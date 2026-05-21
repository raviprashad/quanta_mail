"""
crypto/symmetric.py
───────────────────
AES-256-GCM bulk encryption for message payloads.

AES-256 with Grover's algorithm:
  Grover's quantum search algorithm can search an N-item space in √N steps.
  For AES-256, this reduces effective security from 256 bits to 128 bits.
  128-bit post-quantum security is our target level (matching ML-KEM-768
  category 3), so AES-256 is the correct choice here — not AES-128.

GCM mode provides:
  • Confidentiality   — CTR-mode encryption hides plaintext
  • Integrity         — GHASH authentication tag detects tampering
  • Authenticity      — tag verification requires the correct key

Associated data (AAD):
  We always authenticate the message header as AAD even though it is not
  encrypted.  This ensures the sender's ID, recipient's ID, and sequence
  number cannot be tampered with without invalidating the tag.
"""

from __future__ import annotations

from Crypto.Cipher import AES
from dataclasses import dataclass

import config
from crypto.utils import secure_zero, constant_time_compare


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class EncryptedMessage:
    """
    Output of encrypt_message().

    ciphertext:  Encrypted payload bytes.
    tag:         16-byte GCM authentication tag.
    nonce:       12-byte nonce used for this message (derived from seq no.).
    """
    ciphertext: bytes
    tag: bytes        # 16 bytes
    nonce: bytes      # 12 bytes


# ─── Encryption ───────────────────────────────────────────────────────────────

def encrypt_message(
    key: bytes,
    plaintext: bytes,
    nonce: bytes,
    associated_data: bytes = b"",
) -> EncryptedMessage:
    """
    Encrypt and authenticate a message payload with AES-256-GCM.

    Args:
        key:             32-byte AES-256 encryption key from SessionKeys.
        plaintext:       The message body to encrypt.
        nonce:           12-byte GCM nonce, must be unique per key.
                         Obtain via SessionKeys.derive_nonce(seq_no).
        associated_data: Plaintext data to authenticate but not encrypt
                         (e.g., the message header bytes).

    Returns:
        EncryptedMessage with ciphertext, tag (16 bytes), and nonce.

    Raises:
        ValueError if key or nonce sizes are incorrect.
    """
    _check_key(key)
    _check_nonce(nonce)

    cipher = AES.new(
        key=key,
        mode=AES.MODE_GCM,
        nonce=nonce,
        mac_len=config.GCM_TAG_BYTES,
    )

    if associated_data:
        cipher.update(associated_data)

    ciphertext, tag = cipher.encrypt_and_digest(plaintext)

    return EncryptedMessage(
        ciphertext=ciphertext,
        tag=tag,
        nonce=nonce,
    )


# ─── Decryption ───────────────────────────────────────────────────────────────

def decrypt_message(
    key: bytes,
    encrypted: EncryptedMessage,
    associated_data: bytes = b"",
) -> bytes:
    """
    Decrypt and verify an AES-256-GCM encrypted message.

    The GCM tag is verified in constant time before any plaintext is
    returned.  If verification fails, a ValueError is raised and no
    plaintext is exposed to the caller.

    Args:
        key:             32-byte AES-256 key matching the one used to encrypt.
        encrypted:       EncryptedMessage (ciphertext + tag + nonce).
        associated_data: Must exactly match what was passed at encryption time.

    Returns:
        Decrypted plaintext bytes.

    Raises:
        ValueError   if the authentication tag fails (tampered message).
        ValueError   if key or nonce sizes are wrong.
    """
    _check_key(key)
    _check_nonce(encrypted.nonce)

    if len(encrypted.tag) != config.GCM_TAG_BYTES:
        raise ValueError(
            f"GCM tag must be {config.GCM_TAG_BYTES} bytes, "
            f"got {len(encrypted.tag)}"
        )

    cipher = AES.new(
        key=key,
        mode=AES.MODE_GCM,
        nonce=encrypted.nonce,
        mac_len=config.GCM_TAG_BYTES,
    )

    if associated_data:
        cipher.update(associated_data)

    try:
        plaintext = cipher.decrypt_and_verify(encrypted.ciphertext, encrypted.tag)
    except ValueError:
        # PyCryptodome raises ValueError when the tag does not match.
        # Re-raise with a cleaner message.
        raise ValueError(
            "Message authentication failed. "
            "The message was tampered with, corrupted, or an incorrect key was used."
        )

    return plaintext


# ─── Input validation ─────────────────────────────────────────────────────────

def _check_key(key: bytes) -> None:
    if len(key) != config.SYMMETRIC_KEY_BYTES:
        raise ValueError(
            f"AES key must be {config.SYMMETRIC_KEY_BYTES} bytes (AES-256), "
            f"got {len(key)}"
        )


def _check_nonce(nonce: bytes) -> None:
    if len(nonce) != config.GCM_NONCE_BYTES:
        raise ValueError(
            f"GCM nonce must be {config.GCM_NONCE_BYTES} bytes, "
            f"got {len(nonce)}"
        )
