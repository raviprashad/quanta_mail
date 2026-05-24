"""
protocol/handshake.py
──────────────────────
Post-quantum session establishment protocol.

The handshake establishes a shared session key between two parties
using ML-KEM for key agreement and ML-DSA for mutual authentication.

Full flow:
  ┌──────────────────────────────────────────────────────────────────┐
  │ CLIENT                              SERVER                       │
  │                                                                  │
  │  1. Generate ephemeral KEM keypair                               │
  │  2. Build ClientHello:                                           │
  │     { kem_encap_key, client_cert, timestamp }                   │
  │  3. Sign ClientHello with ML-DSA identity key                    │
  │  4. Send ClientHello ──────────────────────────────────>         │
  │                                                                  │
  │                          5. Verify client cert (SLH-DSA via CA) │
  │                          6. Verify ClientHello signature (ML-DSA)│
  │                          7. Encapsulate with client's KEM key    │
  │                             → shared_secret, ciphertext          │
  │                          8. Build ServerHello:                   │
  │                             { kem_ciphertext, server_cert }      │
  │                          9. Sign transcript + ServerHello        │
  │         <───────────────── 10. Send ServerHello                  │
  │                                                                  │
  │ 11. Verify server cert                                           │
  │ 12. Verify server signature                                      │
  │ 13. Decapsulate ciphertext → shared_secret                      │
  │ 14. Both sides derive session keys via HKDF                      │
  │ 15. Build ClientFinished: { confirmation_tag }                   │
  │     (GCM-encrypted with new session key — proves key agreement) │
  │ 16. Send ClientFinished ───────────────────────────────>         │
  │                                                                  │
  │                         17. Decrypt ClientFinished               │
  │                             → session is established             │
  └──────────────────────────────────────────────────────────────────┘

After the handshake both parties hold identical SessionKeys.
All further messages use those keys via the Session class.
"""

from __future__ import annotations
import time
import struct
from dataclasses import dataclass, field
from typing import Optional, Tuple

import msgpack

import config
from crypto.kem import generate_keypair, encapsulate, decapsulate, KEMKeyPair
from crypto.signing import mldsa_sign, mldsa_verify
from crypto.kdf import derive_session_keys, SessionKeys
from crypto.symmetric import encrypt_message, decrypt_message, EncryptedMessage
from crypto.utils import (
    TranscriptHasher,
    random_bytes,
    secure_zero,
    constant_time_compare,
)
from identity.keypair import UserIdentity
from identity.certificate import Certificate, CertificateAuthority, CertificateError


# ─── Wire message structures ──────────────────────────────────────────────────

@dataclass
class ClientHello:
    """
    First message sent by the client.

    Contains:
      kem_encap_key   — ephemeral ML-KEM-768 encapsulation key
      certificate     — client's CA-signed certificate
      timestamp       — Unix timestamp (replay protection)
      signature       — ML-DSA-65 signature over the above fields
    """
    kem_encap_key: bytes   # 1184 bytes
    certificate:   bytes   # serialised Certificate
    timestamp:     int
    signature:     bytes = b""  # 3309 bytes, filled by client

    def tbs_bytes(self) -> bytes:
        """Bytes that are signed (excludes signature field itself)."""
        return msgpack.packb({
            "type":          "ClientHello",
            "kem_encap_key": self.kem_encap_key,
            "certificate":   self.certificate,
            "timestamp":     self.timestamp,
        }, use_bin_type=True)

    def to_bytes(self) -> bytes:
        return msgpack.packb({
            "type":          "ClientHello",
            "kem_encap_key": self.kem_encap_key,
            "certificate":   self.certificate,
            "timestamp":     self.timestamp,
            "signature":     self.signature,
        }, use_bin_type=True)

    @classmethod
    def from_bytes(cls, data: bytes) -> "ClientHello":
        d = msgpack.unpackb(data, raw=False)
        return cls(
            kem_encap_key=bytes(d["kem_encap_key"]),
            certificate=bytes(d["certificate"]),
            timestamp=d["timestamp"],
            signature=bytes(d["signature"]),
        )


@dataclass
class ServerHello:
    """
    Server's response to ClientHello.

    Contains:
      kem_ciphertext — ML-KEM ciphertext encapsulated with client's key
      certificate    — server's CA-signed certificate
      signature      — ML-DSA signature over transcript + this message
    """
    kem_ciphertext: bytes  # 1088 bytes
    certificate:    bytes  # serialised Certificate
    signature:      bytes = b""  # 3309 bytes

    def tbs_bytes(self, transcript_digest: bytes) -> bytes:
        """Bytes that are signed: transcript so far + this message's fields."""
        return msgpack.packb({
            "type":           "ServerHello",
            "transcript":     transcript_digest,
            "kem_ciphertext": self.kem_ciphertext,
            "certificate":    self.certificate,
        }, use_bin_type=True)

    def to_bytes(self) -> bytes:
        return msgpack.packb({
            "type":           "ServerHello",
            "kem_ciphertext": self.kem_ciphertext,
            "certificate":    self.certificate,
            "signature":      self.signature,
        }, use_bin_type=True)

    @classmethod
    def from_bytes(cls, data: bytes) -> "ServerHello":
        d = msgpack.unpackb(data, raw=False)
        return cls(
            kem_ciphertext=bytes(d["kem_ciphertext"]),
            certificate=bytes(d["certificate"]),
            signature=bytes(d["signature"]),
        )


@dataclass
class ClientFinished:
    """
    Final handshake message from client.

    A small confirmation message encrypted with the newly derived session
    key.  If the server can decrypt it, both sides derived the same key.
    """
    encrypted_confirmation: bytes  # AES-256-GCM encrypted b"finished"
    tag:                    bytes  # 16-byte GCM tag
    nonce:                  bytes  # 12-byte GCM nonce

    def to_bytes(self) -> bytes:
        return msgpack.packb({
            "type":  "ClientFinished",
            "ct":    self.encrypted_confirmation,
            "tag":   self.tag,
            "nonce": self.nonce,
        }, use_bin_type=True)

    @classmethod
    def from_bytes(cls, data: bytes) -> "ClientFinished":
        d = msgpack.unpackb(data, raw=False)
        return cls(
            encrypted_confirmation=bytes(d["ct"]),
            tag=bytes(d["tag"]),
            nonce=bytes(d["nonce"]),
        )


# ─── Established session result ───────────────────────────────────────────────

@dataclass
class EstablishedSession:
    """
    Returned after a successful handshake.

    Contains the session keys and the peer's identity for use by the
    Session class in protocol/session.py.
    """
    client_keys:      SessionKeys    # client-to-server keys
    server_keys:      SessionKeys    # server-to-client keys
    peer_certificate: Certificate    # the remote party's verified certificate
    session_id:       bytes          # 32-byte session identifier


# ─── Client-side handshake ────────────────────────────────────────────────────

class ClientHandshake:
    """
    Manages the client side of the handshake.

    Usage:
        hs = ClientHandshake(client_identity, ca_verifier)
        client_hello_bytes = hs.build_client_hello()
        # send client_hello_bytes to server ...
        # receive server_hello_bytes from server ...
        client_finished_bytes, session = hs.process_server_hello(server_hello_bytes)
        # send client_finished_bytes to server ...
        # session is now ready for use
    """

    def __init__(
        self,
        identity:    UserIdentity,
        ca_verifier: CertificateAuthority,
    ):
        self._identity    = identity
        self._ca_verifier = ca_verifier
        self._transcript  = TranscriptHasher()
        self._kem_keypair: Optional[KEMKeyPair] = None
        self._client_hello_bytes: Optional[bytes] = None

    def build_client_hello(self) -> bytes:
        """
        Build and sign the ClientHello message.

        Generates a fresh ephemeral KEM key pair for this session.
        Signs the ClientHello with the client's long-term ML-DSA key.

        Returns:
            Serialised ClientHello bytes to send to the server.
        """
        if self._identity.certificate is None:
            raise HandshakeError("Client has no certificate. Register with CA first.")

        # Generate ephemeral KEM key pair (new for every session)
        self._kem_keypair = generate_keypair()

        hello = ClientHello(
            kem_encap_key=self._kem_keypair.encap_key,
            certificate=self._identity.certificate.to_bytes(),
            timestamp=int(time.time()),
        )

        # Sign the TBS portion with the client's long-term ML-DSA key
        hello.signature = mldsa_sign(
            private_key=self._identity.mldsa_private_key,
            message=hello.tbs_bytes(),
            context=config.DSA_CTX_HANDSHAKE,
        )

        self._client_hello_bytes = hello.to_bytes()

        # Add to transcript — both sides must hash the same bytes
        self._transcript.add(self._client_hello_bytes)

        return self._client_hello_bytes

    def process_server_hello(
        self, server_hello_bytes: bytes
    ) -> Tuple[bytes, EstablishedSession]:
        """
        Process the ServerHello, derive session keys, build ClientFinished.

        Steps:
          1. Deserialise and verify server's certificate (via CA).
          2. Verify server's ML-DSA signature over the transcript.
          3. Decapsulate the KEM ciphertext → shared_secret.
          4. Derive session keys via HKDF.
          5. Build ClientFinished (proof that we have the correct session key).

        Args:
            server_hello_bytes: Raw bytes received from server.

        Returns:
            Tuple of (client_finished_bytes, EstablishedSession).

        Raises:
            HandshakeError on any verification failure.
        """
        if self._kem_keypair is None:
            raise HandshakeError("build_client_hello() must be called first")

        server_hello = ServerHello.from_bytes(server_hello_bytes)

        # ── Verify server certificate ─────────────────────────────────────────
        server_cert = self._verify_and_parse_cert(server_hello.certificate)

        # ── Compute transcript digest at this point (before adding ServerHello)
        pre_server_transcript = self._transcript.digest()

        # ── Verify server's signature over (transcript + ServerHello fields) ───
        tbs = server_hello.tbs_bytes(pre_server_transcript)
        if not mldsa_verify(
            public_key=server_cert.mldsa_public_key,
            message=tbs,
            signature=server_hello.signature,
            context=config.DSA_CTX_HANDSHAKE,
        ):
            raise HandshakeError("Server signature verification failed")

        # ── Add ServerHello to transcript ─────────────────────────────────────
        self._transcript.add(server_hello_bytes)

        # ── Decapsulate → shared_secret ───────────────────────────────────────
        shared_secret = decapsulate(
            decap_key=self._kem_keypair.decap_key,
            ciphertext=server_hello.kem_ciphertext,
        )

        # ── Destroy ephemeral decap key (forward secrecy) ─────────────────────
        self._kem_keypair.destroy()
        self._kem_keypair = None

        # ── Derive session keys ───────────────────────────────────────────────
        transcript_digest = self._transcript.digest()
        session_id = transcript_digest[:32]

        client_keys = derive_session_keys(
            shared_secret=bytes(shared_secret),
            handshake_transcript=transcript_digest,
            role="client",
        )
        server_keys = derive_session_keys(
            shared_secret=bytes(shared_secret),
            handshake_transcript=transcript_digest,
            role="server",
        )

        # Zero the shared secret — session keys are now derived
        secure_zero(shared_secret)

        # ── Build ClientFinished ──────────────────────────────────────────────
        # Encrypt a fixed confirmation string with the client's enc key
        # to prove we derived the same session key.
        nonce = client_keys.derive_nonce(sequence_number=0)
        enc = encrypt_message(
            key=client_keys.enc_key,
            plaintext=b"handshake-finished-client",
            nonce=nonce,
            associated_data=session_id,
        )

        finished = ClientFinished(
            encrypted_confirmation=enc.ciphertext,
            tag=enc.tag,
            nonce=enc.nonce,
        )
        finished_bytes = finished.to_bytes()
        self._transcript.add(finished_bytes)

        session = EstablishedSession(
            client_keys=client_keys,
            server_keys=server_keys,
            peer_certificate=server_cert,
            session_id=session_id,
        )

        return finished_bytes, session

    def _verify_and_parse_cert(self, cert_bytes: bytes) -> Certificate:
        cert = Certificate.from_bytes(cert_bytes)
        try:
            self._ca_verifier.verify_certificate(cert)
        except CertificateError as e:
            raise HandshakeError(f"Peer certificate invalid: {e}") from e
        return cert


# ─── Server-side handshake ────────────────────────────────────────────────────

class ServerHandshake:
    """
    Manages the server side of the handshake.

    Usage:
        hs = ServerHandshake(server_identity, ca_verifier)
        server_hello_bytes = hs.process_client_hello(client_hello_bytes)
        # send server_hello_bytes to client ...
        # receive client_finished_bytes from client ...
        session = hs.process_client_finished(client_finished_bytes)
        # session is now ready for use
    """

    def __init__(
        self,
        identity:    UserIdentity,
        ca_verifier: CertificateAuthority,
    ):
        self._identity    = identity
        self._ca_verifier = ca_verifier
        self._transcript  = TranscriptHasher()
        self._session:    Optional[EstablishedSession] = None

    def process_client_hello(self, client_hello_bytes: bytes) -> bytes:
        """
        Process ClientHello and build ServerHello.

        Steps:
          1. Deserialise ClientHello.
          2. Verify timestamp (reject replays older than 60 seconds).
          3. Verify client certificate via CA.
          4. Verify client's ML-DSA signature.
          5. Encapsulate with client's KEM key → shared_secret + ciphertext.
          6. Derive session keys.
          7. Sign and return ServerHello.

        Args:
            client_hello_bytes: Raw bytes received from client.

        Returns:
            Serialised ServerHello bytes to send back to client.
        """
        if self._identity.certificate is None:
            raise HandshakeError("Server has no certificate. Register with CA first.")

        # ── Add to transcript ─────────────────────────────────────────────────
        self._transcript.add(client_hello_bytes)

        client_hello = ClientHello.from_bytes(client_hello_bytes)

        # ── Timestamp check (replay protection) ───────────────────────────────
        now = int(time.time())
        skew = abs(now - client_hello.timestamp)
        if skew > 60:
            raise HandshakeError(
                f"ClientHello timestamp skew too large ({skew}s). "
                "Possible replay attack."
            )

        # ── Verify client certificate ─────────────────────────────────────────
        client_cert = self._verify_and_parse_cert(client_hello.certificate)

        # ── Verify client's ML-DSA signature ──────────────────────────────────
        if not mldsa_verify(
            public_key=client_cert.mldsa_public_key,
            message=client_hello.tbs_bytes(),
            signature=client_hello.signature,
            context=config.DSA_CTX_HANDSHAKE,
        ):
            raise HandshakeError("Client signature verification failed")

        # ── Encapsulate with client's KEM key ─────────────────────────────────
        encap_result = encapsulate(client_hello.kem_encap_key)

        # ── Transcript digest before ServerHello ──────────────────────────────
        pre_server_transcript = self._transcript.digest()

        # ── Build ServerHello ─────────────────────────────────────────────────
        server_hello = ServerHello(
            kem_ciphertext=encap_result.ciphertext,
            certificate=self._identity.certificate.to_bytes(),
        )

        # Sign (transcript + ServerHello fields) with server's ML-DSA key
        tbs = server_hello.tbs_bytes(pre_server_transcript)
        server_hello.signature = mldsa_sign(
            private_key=self._identity.mldsa_private_key,
            message=tbs,
            context=config.DSA_CTX_HANDSHAKE,
        )

        server_hello_bytes = server_hello.to_bytes()
        self._transcript.add(server_hello_bytes)

        # ── Derive session keys ───────────────────────────────────────────────
        transcript_digest = self._transcript.digest()
        session_id = transcript_digest[:32]

        client_keys = derive_session_keys(
            shared_secret=bytes(encap_result.shared_secret),
            handshake_transcript=transcript_digest,
            role="client",
        )
        server_keys = derive_session_keys(
            shared_secret=bytes(encap_result.shared_secret),
            handshake_transcript=transcript_digest,
            role="server",
        )

        secure_zero(encap_result.shared_secret)

        self._session = EstablishedSession(
            client_keys=client_keys,
            server_keys=server_keys,
            peer_certificate=client_cert,
            session_id=session_id,
        )

        return server_hello_bytes

    def process_client_finished(
        self, client_finished_bytes: bytes
    ) -> EstablishedSession:
        """
        Verify the ClientFinished message and finalise the session.

        Decrypts the confirmation tag using the derived client session key.
        If decryption succeeds, both sides have the same key — session established.

        Args:
            client_finished_bytes: Raw bytes received from client.

        Returns:
            EstablishedSession ready for message exchange.

        Raises:
            HandshakeError if the confirmation tag is wrong.
        """
        if self._session is None:
            raise HandshakeError("process_client_hello() must be called first")

        finished = ClientFinished.from_bytes(client_finished_bytes)

        enc = EncryptedMessage(
            ciphertext=finished.encrypted_confirmation,
            tag=finished.tag,
            nonce=finished.nonce,
        )

        try:
            plaintext = decrypt_message(
                key=self._session.client_keys.enc_key,
                encrypted=enc,
                associated_data=self._session.session_id,
            )
        except ValueError:
            raise HandshakeError(
                "ClientFinished decryption failed. "
                "Key agreement may have been tampered with."
            )

        if not constant_time_compare(plaintext, b"handshake-finished-client"):
            raise HandshakeError("ClientFinished confirmation value incorrect")

        self._transcript.add(client_finished_bytes)
        return self._session

    def _verify_and_parse_cert(self, cert_bytes: bytes) -> Certificate:
        cert = Certificate.from_bytes(cert_bytes)
        try:
            self._ca_verifier.verify_certificate(cert)
        except CertificateError as e:
            raise HandshakeError(f"Peer certificate invalid: {e}") from e
        return cert


# ─── Exceptions ───────────────────────────────────────────────────────────────

class HandshakeError(Exception):
    """Raised when any step of the handshake fails."""
    pass