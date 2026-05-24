"""
identity/storage.py
────────────────────
Encrypted on-disk storage for user identity key pairs.

Private keys are NEVER stored in plaintext.  This module:
  1. Derives a storage key from the user's password via PBKDF2.
  2. Encrypts all private key bytes with AES-256-GCM.
  3. Stores the encrypted blob + salt + metadata in a JSON file.
  4. On load, derives the same key, decrypts, and reconstructs the identity.

File layout (one JSON file per user):
  {
    "version":       1,
    "user_id":       "alice",
    "display_name":  "Alice Example",
    "salt":          "<hex>",          ← random 32 bytes, stored in plaintext
    "nonce":         "<hex>",          ← AES-GCM nonce
    "tag":           "<hex>",          ← AES-GCM auth tag
    "ciphertext":    "<hex>",          ← encrypted key bundle
    "certificate":   "<hex>" or null   ← serialised certificate (not encrypted)
  }

The certificate is stored unencrypted because it is public information
(it contains only public keys).  Only the private keys are encrypted.

Security note on the password:
  PBKDF2 with 600,000 SHA-256 iterations (NIST SP 800-132) makes
  dictionary attacks expensive.  For production, consider argon2id
  (pip install argon2-cffi) which is more resistant to GPU attacks.
"""

from __future__ import annotations
import os
import json
import struct
from pathlib import Path
from typing import Optional

import msgpack
from Crypto.Cipher import AES

import config
from crypto.kdf import derive_storage_key
from crypto.utils import random_bytes, secure_zero
from identity.keypair import UserIdentity, reconstruct_identity
from identity.certificate import Certificate


# ─── Save ─────────────────────────────────────────────────────────────────────

def save_identity(identity: UserIdentity, password: str) -> Path:
    """
    Encrypt and save a user identity to disk.

    Creates the key storage directory if it does not exist.
    Overwrites any existing file for this user_id.

    Args:
        identity:  The UserIdentity to persist.
        password:  User's passphrase used to derive the encryption key.

    Returns:
        Path to the saved file.
    """
    storage_dir = Path(config.KEY_STORAGE_DIR)
    storage_dir.mkdir(parents=True, exist_ok=True)

    filepath = storage_dir / f"{_safe_filename(identity.user_id)}.json"

    # ── Derive encryption key from password ───────────────────────────────────
    salt = random_bytes(config.KEY_SALT_BYTES)
    storage_key = derive_storage_key(password, salt)

    # ── Serialise all private key material into one bundle ────────────────────
    key_bundle = _pack_key_bundle(identity)

    # ── Encrypt with AES-256-GCM ──────────────────────────────────────────────
    nonce = random_bytes(config.GCM_NONCE_BYTES)
    cipher = AES.new(bytes(storage_key), AES.MODE_GCM, nonce=nonce,
                     mac_len=config.GCM_TAG_BYTES)
    # AAD = user_id prevents ciphertext from being transplanted to another user
    cipher.update(identity.user_id.encode())
    ciphertext, tag = cipher.encrypt_and_digest(key_bundle)

    # ── Zero intermediate secrets ─────────────────────────────────────────────
    secure_zero(storage_key)
    secure_zero(bytearray(key_bundle))

    # ── Build the on-disk record ──────────────────────────────────────────────
    cert_hex = None
    if identity.certificate:
        cert_hex = identity.certificate.to_bytes().hex()

    record = {
        "version":      1,
        "user_id":      identity.user_id,
        "display_name": identity.display_name,
        "salt":         salt.hex(),
        "nonce":        nonce.hex(),
        "tag":          tag.hex(),
        "ciphertext":   ciphertext.hex(),
        "certificate":  cert_hex,
    }

    filepath.write_text(json.dumps(record, indent=2))
    return filepath


# ─── Load ─────────────────────────────────────────────────────────────────────

def load_identity(user_id: str, password: str) -> UserIdentity:
    """
    Load and decrypt a user identity from disk.

    Args:
        user_id:   Must match the user_id used when saving.
        password:  Same passphrase used when saving.

    Returns:
        Fully reconstructed UserIdentity.

    Raises:
        FileNotFoundError  if no key file exists for this user_id.
        ValueError         if the password is wrong or the file is corrupted.
    """
    filepath = Path(config.KEY_STORAGE_DIR) / f"{_safe_filename(user_id)}.json"

    if not filepath.exists():
        raise FileNotFoundError(
            f"No key file found for user '{user_id}' at {filepath}"
        )

    record = json.loads(filepath.read_text())

    if record.get("version") != 1:
        raise ValueError(f"Unsupported key file version: {record.get('version')}")
    if record["user_id"] != user_id:
        raise ValueError("user_id in file does not match requested user_id")

    # ── Reconstruct encryption key from password ──────────────────────────────
    salt = bytes.fromhex(record["salt"])
    storage_key = derive_storage_key(password, salt)

    # ── Decrypt ───────────────────────────────────────────────────────────────
    nonce      = bytes.fromhex(record["nonce"])
    tag        = bytes.fromhex(record["tag"])
    ciphertext = bytes.fromhex(record["ciphertext"])

    cipher = AES.new(bytes(storage_key), AES.MODE_GCM, nonce=nonce,
                     mac_len=config.GCM_TAG_BYTES)
    cipher.update(user_id.encode())

    try:
        key_bundle = cipher.decrypt_and_verify(ciphertext, tag)
    except ValueError:
        secure_zero(storage_key)
        raise ValueError(
            "Decryption failed. Wrong password or corrupted key file."
        )
    finally:
        secure_zero(storage_key)

    # ── Unpack key bundle ─────────────────────────────────────────────────────
    (
        mldsa_public_key,
        mldsa_private_key,
        mlkem_encap_key,
        mlkem_decap_key,
    ) = _unpack_key_bundle(key_bundle)

    secure_zero(bytearray(key_bundle))

    # ── Deserialise certificate if present ────────────────────────────────────
    certificate = None
    if record.get("certificate"):
        certificate = Certificate.from_bytes(bytes.fromhex(record["certificate"]))

    return reconstruct_identity(
        user_id=record["user_id"],
        display_name=record["display_name"],
        mldsa_public_key=mldsa_public_key,
        mldsa_private_key=mldsa_private_key,
        mlkem_encap_key=mlkem_encap_key,
        mlkem_decap_key=mlkem_decap_key,
        certificate=certificate,
    )


def update_certificate(user_id: str, certificate: Certificate) -> None:
    """
    Update the stored certificate for a user without re-encrypting keys.

    Called after the CA issues a new or renewed certificate.
    The key ciphertext is not touched — only the certificate field is updated.

    Args:
        user_id:     Must match an existing key file.
        certificate: The new CA-signed certificate.
    """
    filepath = Path(config.KEY_STORAGE_DIR) / f"{_safe_filename(user_id)}.json"

    if not filepath.exists():
        raise FileNotFoundError(f"No key file for user '{user_id}'")

    record = json.loads(filepath.read_text())
    record["certificate"] = certificate.to_bytes().hex()
    filepath.write_text(json.dumps(record, indent=2))


def identity_exists(user_id: str) -> bool:
    """Return True if a key file exists for this user_id."""
    filepath = Path(config.KEY_STORAGE_DIR) / f"{_safe_filename(user_id)}.json"
    return filepath.exists()


# ─── Serialisation helpers ────────────────────────────────────────────────────

def _pack_key_bundle(identity: UserIdentity) -> bytes:
    """
    Pack all four key byte strings into a single msgpack blob.
    Only this blob is encrypted — structure is deterministic.
    """
    bundle = {
        "mldsa_pk": identity.mldsa_public_key,
        "mldsa_sk": identity.mldsa_private_key,
        "mlkem_ek": identity.mlkem_encap_key,
        "mlkem_dk": identity.mlkem_decap_key,
    }
    return msgpack.packb(bundle, use_bin_type=True)


def _unpack_key_bundle(
    bundle_bytes: bytes,
) -> tuple[bytes, bytes, bytes, bytes]:
    """Unpack the key bundle and return (mldsa_pk, mldsa_sk, mlkem_ek, mlkem_dk)."""
    d = msgpack.unpackb(bundle_bytes, raw=False)
    return (
        bytes(d["mldsa_pk"]),
        bytes(d["mldsa_sk"]),
        bytes(d["mlkem_ek"]),
        bytes(d["mlkem_dk"]),
    )


def _safe_filename(user_id: str) -> str:
    """Strip non-alphanumeric characters for safe filenames."""
    return "".join(c for c in user_id if c.isalnum() or c in "-_")