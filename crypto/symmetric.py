"""
crypto/symmetric.py
────────────────────
AES-256-GCM authenticated encryption for message payloads.

This module provides the lowest-level encrypt/decrypt primitives used by
the session layer (protocol/session.py) and the handshake confirmation
step (protocol/handshake.py).

Why AES-256-GCM?
  • AES-256 gives 128-bit post-quantum security after Grover's halving.
  • GCM provides authenticated encryption with associated data (AEAD),
    so we can bind plaintext headers to the ciphertext without encrypting
    them — the server can route on headers without decrypting the body,
    but any tampering with headers is detected.
  • NIST SP 800-38D specifies GCM; it is FIPS 197 approved.

Nonce policy:
  Callers MUST supply a unique 12-byte nonce per (key, message) pair.
  The session layer derives nonces deterministically from a sequence
  number via HMAC-SHA3-256 (see crypto/kdf.py SessionKeys.derive_nonce).
  Never reuse a nonce with the same key — AES-GCM is catastrophically
  broken under nonce reuse.

Dependencies:
  pycryptodome (PyCryptodome) — already in requirements.txt.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from Crypto.Cipher import AES

import config


# ─── Data class ───────────────────────────────────────────────────────────────

@dataclass
class EncryptedMessage:
    """
    Output of AES-256-GCM encryption.

    Attributes:
        ciphertext — encrypted payload (same length as plaintext)
        tag        — 16-byte GCM authentication tag
        nonce      — 12-byte nonce used for this encryption
    """
    ciphertext: bytes
    tag:        bytes
    nonce:      bytes


# ─── Encrypt ──────────────────────────────────────────────────────────────────

def encrypt_message(
    key:              bytes,
    plaintext:        bytes,
    nonce:            bytes,
    associated_data:  Optional[bytes] = None,
) -> EncryptedMessage:
    """
    Encrypt a plaintext payload with AES-256-GCM.

    Args:
        key:              32-byte AES-256 encryption key.
        plaintext:        Arbitrary-length bytes to encrypt.
        nonce:            12-byte nonce (MUST be unique per key).
        associated_data:  Optional additional authenticated data (AAD).
                          The AAD is covered by the GCM tag but is NOT
                          encrypted — use it for plaintext headers that
                          must be tamper-proof but visible for routing.

    Returns:
        EncryptedMessage containing ciphertext, tag, and nonce.

    Raises:
        ValueError if key or nonce sizes are incorrect.
    """
    _validate_key(key)
    _validate_nonce(nonce)

    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce, mac_len=config.GCM_TAG_BYTES)

    if associated_data is not None:
        cipher.update(associated_data)

    ciphertext, tag = cipher.encrypt_and_digest(plaintext)

    return EncryptedMessage(
        ciphertext=ciphertext,
        tag=tag,
        nonce=nonce,
    )


# ─── Decrypt ──────────────────────────────────────────────────────────────────

def decrypt_message(
    key:              bytes,
    encrypted:        EncryptedMessage,
    associated_data:  Optional[bytes] = None,
) -> bytes:
    """
    Decrypt and verify an AES-256-GCM encrypted message.

    Verification covers both the ciphertext and the associated data.
    If any byte of either has been tampered with, the GCM tag check
    fails and a ValueError is raised.

    Args:
        key:              32-byte AES-256 encryption key (same as used
                          for encryption).
        encrypted:        EncryptedMessage produced by encrypt_message().
        associated_data:  Must match the AAD supplied at encryption time.
                          Pass None if no AAD was used.

    Returns:
        Decrypted plaintext bytes.

    Raises:
        ValueError if the authentication tag is invalid (tampered data,
                   wrong key, or wrong AAD) or if key/nonce sizes are
                   incorrect.
    """
    _validate_key(key)
    _validate_nonce(encrypted.nonce)

    cipher = AES.new(
        key, AES.MODE_GCM, nonce=encrypted.nonce, mac_len=config.GCM_TAG_BYTES,
    )

    if associated_data is not None:
        cipher.update(associated_data)

    try:
        plaintext = cipher.decrypt_and_verify(encrypted.ciphertext, encrypted.tag)
    except ValueError:
        raise ValueError(
            "GCM authentication failed. The ciphertext, tag, associated "
            "data, or key is incorrect. Possible tampering detected."
        )

    return plaintext


# ─── Validation helpers ───────────────────────────────────────────────────────

def _validate_key(key: bytes) -> None:
    """Ensure the AES key is exactly 32 bytes (256 bits)."""
    if len(key) != config.SYMMETRIC_KEY_BYTES:
        raise ValueError(
            f"AES-256 key must be {config.SYMMETRIC_KEY_BYTES} bytes, "
            f"got {len(key)}"
        )


def _validate_nonce(nonce: bytes) -> None:
    """Ensure the GCM nonce is exactly 12 bytes (96 bits)."""
    if len(nonce) != config.GCM_NONCE_BYTES:
        raise ValueError(
            f"GCM nonce must be {config.GCM_NONCE_BYTES} bytes, "
            f"got {len(nonce)}"
        )