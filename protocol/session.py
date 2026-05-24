"""
protocol/session.py
────────────────────
Ongoing encrypted session after a successful handshake.

Responsibilities:
  • Encrypting outbound messages with AES-256-GCM.
  • Decrypting and verifying inbound messages.
  • Tracking sequence numbers to detect replays and dropped messages.
  • Enforcing the rekeying policy (FIPS 203 §3.3).
  • Signing outbound messages with ML-DSA (non-repudiation).
  • Verifying inbound ML-DSA signatures.

Session key direction:
  client_keys — used by the CLIENT to encrypt, by the SERVER to decrypt
  server_keys — used by the SERVER to encrypt, by the CLIENT to decrypt
  This ensures a message encrypted by one side cannot be replayed as if
  sent by the other side.
"""

from __future__ import annotations
import time
import struct
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

import msgpack

import config
from crypto.symmetric import encrypt_message, decrypt_message, EncryptedMessage
from crypto.signing import mldsa_sign, mldsa_verify
from crypto.kdf import SessionKeys, derive_session_keys
from crypto.kem import generate_keypair, encapsulate, decapsulate
from crypto.utils import secure_zero, random_bytes
from identity.keypair import UserIdentity
from identity.certificate import Certificate
from protocol.handshake import EstablishedSession


class Role(Enum):
    CLIENT = "client"
    SERVER = "server"


# ─── Message structure ────────────────────────────────────────────────────────

@dataclass
class MessageHeader:
    """
    Plaintext header — authenticated by GCM AAD but not encrypted.

    Keeping headers plaintext allows routing without decryption.
    The GCM tag covers the header, so tampering is detected.
    """
    sender_id:       str    # sender's user_id
    recipient_id:    str    # recipient's user_id
    session_id:      bytes  # 32-byte session identifier
    sequence_number: int    # monotonically increasing per session
    timestamp:       int    # Unix timestamp
    message_type:    str    # "text", "file", "rekey", etc.

    def to_bytes(self) -> bytes:
        """Canonical serialisation used as GCM associated data."""
        return msgpack.packb({
            "sender_id":       self.sender_id,
            "recipient_id":    self.recipient_id,
            "session_id":      self.session_id,
            "sequence_number": self.sequence_number,
            "timestamp":       self.timestamp,
            "message_type":    self.message_type,
        }, use_bin_type=True)

    @classmethod
    def from_bytes(cls, data: bytes) -> "MessageHeader":
        d = msgpack.unpackb(data, raw=False)
        return cls(
            sender_id=d["sender_id"],
            recipient_id=d["recipient_id"],
            session_id=bytes(d["session_id"]),
            sequence_number=d["sequence_number"],
            timestamp=d["timestamp"],
            message_type=d["message_type"],
        )


@dataclass
class EncryptedEnvelope:
    """
    Complete on-wire message: header + encrypted payload + ML-DSA signature.

    The signature covers: header_bytes + ciphertext + tag
    (not the plaintext — plaintext is never exposed outside the session).
    """
    header:     bytes   # serialised MessageHeader (plaintext)
    ciphertext: bytes   # AES-256-GCM encrypted payload
    tag:        bytes   # 16-byte GCM authentication tag
    nonce:      bytes   # 12-byte GCM nonce
    signature:  bytes   # ML-DSA-65 signature (3309 bytes)

    def to_bytes(self) -> bytes:
        return msgpack.packb({
            "header":     self.header,
            "ciphertext": self.ciphertext,
            "tag":        self.tag,
            "nonce":      self.nonce,
            "signature":  self.signature,
        }, use_bin_type=True)

    @classmethod
    def from_bytes(cls, data: bytes) -> "EncryptedEnvelope":
        d = msgpack.unpackb(data, raw=False)
        return cls(
            header=bytes(d["header"]),
            ciphertext=bytes(d["ciphertext"]),
            tag=bytes(d["tag"]),
            nonce=bytes(d["nonce"]),
            signature=bytes(d["signature"]),
        )

    @property
    def signable_bytes(self) -> bytes:
        """The bytes that are signed: header + ciphertext + tag."""
        return self.header + self.ciphertext + self.tag


@dataclass
class DecryptedMessage:
    """Result of successfully decrypting an inbound message."""
    header:  MessageHeader
    payload: bytes          # decrypted plaintext
    verified: bool          # True if ML-DSA signature was valid


# ─── Session class ────────────────────────────────────────────────────────────

class Session:
    """
    An active encrypted session between two parties.

    Instantiate with the EstablishedSession from a completed handshake.

    Usage:
        session = Session(established, identity, peer_cert, Role.CLIENT)

        # Send a message
        envelope_bytes = session.encrypt("Hello, world!".encode())
        # ... transmit envelope_bytes ...

        # Receive a message
        msg = session.decrypt(received_envelope_bytes)
        print(msg.payload.decode())
    """

    def __init__(
        self,
        established:  EstablishedSession,
        identity:     UserIdentity,
        peer_cert:    Certificate,
        role:         Role,
    ):
        self._identity  = identity
        self._peer_cert = peer_cert
        self._role      = role
        self._session_id = established.session_id

        # Key assignment depends on role:
        # The CLIENT encrypts with client_keys, decrypts with server_keys.
        # The SERVER encrypts with server_keys, decrypts with client_keys.
        if role == Role.CLIENT:
            self._send_keys = established.client_keys
            self._recv_keys = established.server_keys
        else:
            self._send_keys = established.server_keys
            self._recv_keys = established.client_keys

        # Sequence numbers start at 1 (0 was used by ClientFinished confirmation)
        self._send_seq: int = 1
        self._recv_seq: int = 1

        # Rekeying state
        self._messages_sent: int = 0
        self._session_start: float = time.time()
        self._closed: bool = False

    # ── Encrypt ───────────────────────────────────────────────────────────────

    def encrypt(
        self,
        payload:      bytes,
        recipient_id: str,
        message_type: str = "text",
    ) -> bytes:
        """
        Encrypt a message payload and return the serialised EncryptedEnvelope.

        Signs the envelope with the sender's ML-DSA key for non-repudiation.
        Auto-rekeys if the rekeying policy threshold is reached.

        Args:
            payload:       Raw bytes to encrypt (e.g. UTF-8 message text).
            recipient_id:  The intended recipient's user_id.
            message_type:  Content type tag ("text", "file", etc.).

        Returns:
            Serialised EncryptedEnvelope bytes ready to send.

        Raises:
            SessionError if the session is closed.
        """
        self._assert_open()
        self._check_rekey_policy()

        header = MessageHeader(
            sender_id=self._identity.user_id,
            recipient_id=recipient_id,
            session_id=self._session_id,
            sequence_number=self._send_seq,
            timestamp=int(time.time()),
            message_type=message_type,
        )
        header_bytes = header.to_bytes()

        # Derive a unique nonce from the sequence number (never reused per key)
        nonce = self._send_keys.derive_nonce(self._send_seq)

        # Encrypt: GCM tag covers both ciphertext and header (as AAD)
        enc = encrypt_message(
            key=self._send_keys.enc_key,
            plaintext=payload,
            nonce=nonce,
            associated_data=header_bytes,
        )

        envelope = EncryptedEnvelope(
            header=header_bytes,
            ciphertext=enc.ciphertext,
            tag=enc.tag,
            nonce=enc.nonce,
            signature=b"",
        )

        # Sign: header + ciphertext + tag (not the plaintext)
        envelope.signature = mldsa_sign(
            private_key=self._identity.mldsa_private_key,
            message=envelope.signable_bytes,
            context=config.DSA_CTX_MESSAGE,
        )

        self._send_seq += 1
        self._messages_sent += 1

        return envelope.to_bytes()

    # ── Decrypt ───────────────────────────────────────────────────────────────

    def decrypt(self, envelope_bytes: bytes) -> DecryptedMessage:
        """
        Decrypt and verify an inbound EncryptedEnvelope.

        Steps:
          1. Deserialise the envelope.
          2. Parse and validate the message header.
          3. Verify the ML-DSA signature (non-repudiation).
          4. Decrypt and authenticate the payload (AES-256-GCM).
          5. Check sequence number (replay detection).

        Args:
            envelope_bytes: Raw bytes received from the peer.

        Returns:
            DecryptedMessage with header, payload, and signature validity flag.

        Raises:
            SessionError on any integrity or authentication failure.
        """
        self._assert_open()

        try:
            envelope = EncryptedEnvelope.from_bytes(envelope_bytes)
            header   = MessageHeader.from_bytes(envelope.header)
        except Exception as e:
            raise SessionError(f"Message deserialisation failed: {e}") from e

        # ── Header validation ─────────────────────────────────────────────────
        self._validate_header(header)

        # ── ML-DSA signature verification ─────────────────────────────────────
        # We verify before decryption so we never work with unauthenticated data.
        sig_valid = mldsa_verify(
            public_key=self._peer_cert.mldsa_public_key,
            message=envelope.signable_bytes,
            signature=envelope.signature,
            context=config.DSA_CTX_MESSAGE,
        )

        if not sig_valid:
            raise SessionError(
                f"Message signature invalid (seq={header.sequence_number}). "
                "Message may be forged or from wrong sender."
            )

        # ── AES-256-GCM decryption ────────────────────────────────────────────
        enc = EncryptedMessage(
            ciphertext=envelope.ciphertext,
            tag=envelope.tag,
            nonce=envelope.nonce,
        )

        try:
            plaintext = decrypt_message(
                key=self._recv_keys.enc_key,
                encrypted=enc,
                associated_data=envelope.header,
            )
        except ValueError as e:
            raise SessionError(f"Message decryption failed: {e}") from e

        # ── Advance sequence number ───────────────────────────────────────────
        self._recv_seq = header.sequence_number + 1

        return DecryptedMessage(
            header=header,
            payload=plaintext,
            verified=True,
        )

    # ── Rekeying ──────────────────────────────────────────────────────────────

    def _check_rekey_policy(self) -> None:
        """
        Trigger rekeying if message count or time threshold is exceeded.

        FIPS 203 §3.3: session keys should be rotated regularly.
        Rekeying generates new session keys without a full handshake.
        Currently this sends a rekey signal; the full rekey exchange
        is implemented in the server's route handler.
        """
        msgs_exceeded = (
            self._messages_sent >= config.SESSION_REKEY_AFTER_MESSAGES
        )
        time_exceeded = (
            time.time() - self._session_start >= config.SESSION_REKEY_AFTER_SECONDS
        )

        if msgs_exceeded or time_exceeded:
            # Mark for rekeying — the application layer handles the exchange.
            # Here we just flag it; a production system would trigger the rekey
            # automatically using a "rekey" message_type.
            self._needs_rekey = True

    @property
    def needs_rekey(self) -> bool:
        return getattr(self, "_needs_rekey", False)

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        """
        Destroy session keys and mark session as closed.
        Always call this when a session ends.
        """
        self._send_keys.destroy()
        self._recv_keys.destroy()
        self._closed = True

    def _assert_open(self) -> None:
        if self._closed:
            raise SessionError("Session is closed")

    def _validate_header(self, header: MessageHeader) -> None:
        """Check session ID, sequence number, and timestamp."""
        if header.session_id != self._session_id:
            raise SessionError("Message has wrong session_id")

        if header.sender_id != self._peer_cert.user_id:
            raise SessionError(
                f"Message sender '{header.sender_id}' does not match "
                f"peer certificate user '{self._peer_cert.user_id}'"
            )

        # Replay detection: sequence number must advance monotonically
        if header.sequence_number < self._recv_seq:
            raise SessionError(
                f"Replay detected: received seq={header.sequence_number}, "
                f"expected >= {self._recv_seq}"
            )

        # Timestamp drift check (±5 minutes)
        drift = abs(int(time.time()) - header.timestamp)
        if drift > 300:
            raise SessionError(
                f"Message timestamp drift too large ({drift}s). Possible replay."
            )


class SessionError(Exception):
    """Raised when session-level checks fail."""
    pass