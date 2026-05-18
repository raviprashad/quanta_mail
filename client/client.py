"""
client/client.py
─────────────────
High-level client library for PQ Messenger.

Wraps all cryptographic operations and HTTP calls into a simple API:

    client = PQMessengerClient(server_url="http://127.0.0.1:8000")

    # First time: register
    client.register(user_id="alice", display_name="Alice", password="s3cr3t")

    # Subsequent runs: load from storage
    client.load_identity(user_id="alice", password="s3cr3t")

    # Connect (runs full PQ handshake)
    client.connect()

    # Send a message
    client.send("bob", "Hello Bob!")

    # Receive messages
    for msg in client.receive():
        print(f"From {msg.header.sender_id}: {msg.payload.decode()}")

    # Always close when done
    client.close()
"""

from __future__ import annotations
import time
from typing import List, Optional, Iterator

import httpx

import config
from identity.keypair import generate_user_identity, UserIdentity
from identity.storage import save_identity, load_identity, update_certificate
from identity.certificate import Certificate, CertificateAuthority, CertificateError
from protocol.handshake import ClientHandshake, HandshakeError, EstablishedSession
from protocol.session import Session, Role, DecryptedMessage, SessionError


class PQMessengerClient:
    """
    Client for the PQ Messenger server.

    All cryptography happens inside this class.  The caller never
    touches key bytes directly.
    """

    def __init__(self, server_url: str = "http://127.0.0.1:8000"):
        self._server_url  = server_url.rstrip("/")
        self._http        = httpx.Client(timeout=30.0)
        self._identity:   Optional[UserIdentity] = None
        self._ca_pk:      Optional[bytes] = None
        self._session:    Optional[Session] = None
        self._session_id: Optional[str] = None

    # ─── Identity management ──────────────────────────────────────────────────

    def register(
        self,
        user_id:      str,
        display_name: str,
        password:     str,
    ) -> None:
        """
        Generate fresh keys, register with the server, save identity to disk.

        Call this ONCE on first use.  Subsequent runs call load_identity().

        Steps:
          1. Generate ML-DSA-65 and ML-KEM-768 key pairs.
          2. POST public keys to /register.
          3. Server issues an SLH-DSA-signed certificate.
          4. Attach certificate to identity.
          5. Save encrypted identity to disk.

        Args:
            user_id:      Unique identifier (alphanumeric, no spaces).
            display_name: Human-readable name.
            password:     Passphrase to encrypt stored keys.
        """
        print(f"[client] Generating key pairs for '{user_id}' ...")
        identity = generate_user_identity(user_id, display_name)

        print(f"[client] Registering with server ...")
        resp = self._post("/register", {
            "user_id":          user_id,
            "display_name":     display_name,
            "mldsa_public_key": identity.mldsa_public_key.hex(),
            "mlkem_encap_key":  identity.mlkem_encap_key.hex(),
        })

        cert_bytes   = bytes.fromhex(resp["certificate"])
        ca_pk_bytes  = bytes.fromhex(resp["ca_public_key"])

        # Verify the certificate before trusting it
        ca_verifier = CertificateAuthority.verifier_only(ca_pk_bytes)
        cert = Certificate.from_bytes(cert_bytes)
        try:
            ca_verifier.verify_certificate(cert)
        except CertificateError as e:
            identity.destroy()
            raise RuntimeError(f"Server issued invalid certificate: {e}") from e

        identity.attach_certificate(cert)

        # Save everything encrypted to disk
        path = save_identity(identity, password)
        print(f"[client] Identity saved to {path}")
        print(f"[client] Certificate fingerprint: {cert.fingerprint}")

        self._identity = identity
        self._ca_pk    = ca_pk_bytes

    def load_identity(self, user_id: str, password: str) -> None:
        """
        Load a previously registered identity from disk.

        Args:
            user_id:  Must match the user_id used during register().
            password: Same passphrase used during register().

        Raises:
            FileNotFoundError if no key file exists for this user_id.
            ValueError if the password is wrong.
        """
        print(f"[client] Loading identity for '{user_id}' ...")
        self._identity = load_identity(user_id, password)

        # Fetch the CA public key from the server
        resp = self._get("/ca-public-key")
        self._ca_pk = bytes.fromhex(resp["ca_public_key"])

        print(f"[client] Identity loaded (fingerprint: {self._identity.fingerprint})")

    # ─── Connection (handshake) ───────────────────────────────────────────────

    def connect(self) -> None:
        """
        Run the full post-quantum handshake with the server.

        After this method returns, the client has a shared session key with
        the server and all subsequent messages are encrypted with it.

        Steps:
          1. Build and sign ClientHello.
          2. POST to /handshake/hello → receive ServerHello.
          3. Verify server certificate and signature.
          4. Decapsulate KEM ciphertext → shared_secret.
          5. Derive session keys.
          6. Send ClientFinished (proves key agreement).
          7. Receive session_id confirming server accepted the session.
        """
        if self._identity is None:
            raise RuntimeError("Call register() or load_identity() first")
        if self._identity.certificate is None:
            raise RuntimeError("Identity has no certificate. Call register() first.")

        print("[client] Running post-quantum handshake ...")

        ca_verifier = CertificateAuthority.verifier_only(self._ca_pk)
        hs = ClientHandshake(
            identity=self._identity,
            ca_verifier=ca_verifier,
        )

        # ── Step 1–2: ClientHello ─────────────────────────────────────────────
        client_hello_bytes = hs.build_client_hello()
        resp = self._post("/handshake/hello", {
            "client_hello": client_hello_bytes.hex(),
        })

        handshake_id      = resp["handshake_id"]
        server_hello_bytes = bytes.fromhex(resp["server_hello"])

        # ── Step 3–6: Process ServerHello, derive keys, build ClientFinished ──
        try:
            client_finished_bytes, established = hs.process_server_hello(
                server_hello_bytes
            )
        except HandshakeError as e:
            raise RuntimeError(f"Handshake failed: {e}") from e

        # ── Step 6–7: Send ClientFinished ─────────────────────────────────────
        resp = self._post("/handshake/finish", {
            "handshake_id":    handshake_id,
            "client_finished": client_finished_bytes.hex(),
        })

        self._session_id = resp["session_id"]
        self._session = Session(
            established=established,
            identity=self._identity,
            peer_cert=established.peer_certificate,
            role=Role.CLIENT,
        )

        print(f"[client] Session established (id: {self._session_id[:16]}...)")

    # ─── Messaging ────────────────────────────────────────────────────────────

    def send(self, recipient_id: str, message: str) -> None:
        """
        Encrypt and send a text message to recipient_id.

        The message is:
          1. Encrypted with AES-256-GCM using the session key.
          2. Signed with the sender's ML-DSA-65 key.
          3. Transmitted to the server for routing.

        Args:
            recipient_id: The user_id of the intended recipient.
            message:      Plaintext message string.
        """
        if self._session is None:
            raise RuntimeError("Call connect() before sending messages")

        payload       = message.encode("utf-8")
        envelope_bytes = self._session.encrypt(
            payload=payload,
            recipient_id=recipient_id,
            message_type="text",
        )

        self._post(f"/message/{self._session_id}", {
            "envelope":     envelope_bytes.hex(),
            "recipient_id": recipient_id,
        })

        print(f"[client] Message sent to '{recipient_id}'")

    def receive(self) -> List[DecryptedMessage]:
        """
        Poll the server for inbound messages and decrypt them.

        Returns:
            List of DecryptedMessage objects (may be empty if no messages).
        """
        if self._identity is None:
            raise RuntimeError("Not logged in")

        resp = self._get(f"/message/{self._identity.user_id}")
        messages = []

        for env_hex in resp.get("messages", []):
            envelope_bytes = bytes.fromhex(env_hex)
            try:
                # To decrypt inbound messages we need a session keyed for
                # the sender.  In a full system each conversation has its
                # own session.  Here we attempt to decrypt with the current
                # active session as a demonstration.
                if self._session is not None:
                    msg = self._session.decrypt(envelope_bytes)
                    messages.append(msg)
                else:
                    print("[client] Received message but no active session to decrypt")
            except SessionError as e:
                print(f"[client] Warning: could not decrypt message: {e}")

        return messages

    # ─── Cleanup ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        """
        Close the session and destroy all key material.

        Always call this when you are done.
        """
        if self._session is not None and self._session_id is not None:
            try:
                self._http.delete(f"{self._server_url}/session/{self._session_id}")
            except Exception:
                pass  # best effort
            self._session.close()
            self._session = None

        if self._identity is not None:
            self._identity.destroy()
            self._identity = None

        self._http.close()
        print("[client] Session closed and keys destroyed")

    # ─── HTTP helpers ─────────────────────────────────────────────────────────

    def _post(self, path: str, body: dict) -> dict:
        url = self._server_url + path
        try:
            resp = self._http.post(url, json=body)
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to server at {url}. "
                "Is the server running? (uvicorn server.app:app --reload)"
            )
        if not resp.is_success:
            raise RuntimeError(
                f"Server error {resp.status_code} at {path}: {resp.text}"
            )
        return resp.json()

    def _get(self, path: str) -> dict:
        url = self._server_url + path
        try:
            resp = self._http.get(url)
        except httpx.ConnectError:
            raise RuntimeError(f"Cannot connect to server at {url}")
        if not resp.is_success:
            raise RuntimeError(
                f"Server error {resp.status_code} at {path}: {resp.text}"
            )
        return resp.json()