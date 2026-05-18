"""
identity/keypair.py
────────────────────
User identity key pair management.

Each user has a long-term identity consisting of:
  • ML-DSA-65  key pair  — for signing messages and handshake transcripts
  • ML-KEM-768 key pair  — for receiving encrypted session initiations
  • A certificate        — binding identity to those public keys

This module handles generating, loading, and retiring identity keys.
The actual on-disk storage (encrypted) lives in identity/storage.py.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from crypto.signing import mldsa_generate_keypair, MLDSAKeyPair
from crypto.kem import generate_keypair, KEMKeyPair
from crypto.utils import public_key_fingerprint
from identity.certificate import Certificate


@dataclass
class UserIdentity:
    """
    Complete identity for one user.

    Holds both key pairs and the certificate that a CA has signed.
    The certificate is what you send to remote parties.
    The private keys never leave this object.

    Attributes:
        user_id        — unique string identifier
        display_name   — human-readable name shown in UI
        mldsa_keypair  — ML-DSA-65 key pair (sign / verify)
        mlkem_keypair  — ML-KEM-768 key pair (encap / decap)
        certificate    — CA-signed certificate (may be None before enrolment)
    """
    user_id:       str
    display_name:  str
    mldsa_keypair: MLDSAKeyPair
    mlkem_keypair: KEMKeyPair
    certificate:   Optional[Certificate] = None

    @property
    def mldsa_public_key(self) -> bytes:
        return self.mldsa_keypair.public_key

    @property
    def mldsa_private_key(self) -> bytes:
        return self.mldsa_keypair.private_key

    @property
    def mlkem_encap_key(self) -> bytes:
        """Public encapsulation key — share this so others can send you secrets."""
        return self.mlkem_keypair.encap_key

    @property
    def mlkem_decap_key(self) -> bytes:
        """Private decapsulation key — never share this."""
        return self.mlkem_keypair.decap_key

    @property
    def fingerprint(self) -> str:
        return public_key_fingerprint(self.mldsa_public_key)

    def destroy(self) -> None:
        """
        Zero all private key material.
        Call when rotating keys or shutting down.
        FIPS 203 §3.3 / FIPS 204 §3.6.3 require destroying secret data
        when it is no longer needed.
        """
        self.mldsa_keypair.destroy()
        self.mlkem_keypair.destroy()

    def attach_certificate(self, cert: Certificate) -> None:
        """
        Attach a CA-signed certificate to this identity.
        Called after registration with the CA.
        """
        if cert.user_id != self.user_id:
            raise ValueError(
                f"Certificate user_id '{cert.user_id}' does not match "
                f"identity user_id '{self.user_id}'"
            )
        if cert.mldsa_public_key != self.mldsa_public_key:
            raise ValueError(
                "Certificate ML-DSA public key does not match identity key pair"
            )
        if cert.mlkem_encap_key != self.mlkem_encap_key:
            raise ValueError(
                "Certificate ML-KEM encap key does not match identity key pair"
            )
        self.certificate = cert


def generate_user_identity(user_id: str, display_name: str) -> UserIdentity:
    """
    Generate a brand-new user identity with fresh key pairs.

    This should be called once per user on first setup.
    The resulting identity should immediately be:
      1. Saved to encrypted storage (identity/storage.py).
      2. Submitted to the CA for certificate issuance.

    Args:
        user_id:      Unique identifier (e.g. UUID or username).
        display_name: Human-readable name.

    Returns:
        UserIdentity with freshly generated ML-DSA and ML-KEM key pairs.
    """
    mldsa_keypair = mldsa_generate_keypair()
    mlkem_keypair = generate_keypair()

    return UserIdentity(
        user_id=user_id,
        display_name=display_name,
        mldsa_keypair=mldsa_keypair,
        mlkem_keypair=mlkem_keypair,
    )


def reconstruct_identity(
    user_id: str,
    display_name: str,
    mldsa_public_key: bytes,
    mldsa_private_key: bytes,
    mlkem_encap_key: bytes,
    mlkem_decap_key: bytes,
    certificate: Optional[Certificate] = None,
) -> UserIdentity:
    """
    Reconstruct a UserIdentity from raw key bytes.

    Used by the storage layer after loading and decrypting keys from disk.
    Validates all key sizes before returning.

    Args:
        All raw key bytes, plus optional certificate.

    Returns:
        Fully reconstructed UserIdentity.
    """
    import config

    if len(mldsa_public_key) != config.DSA_PUBLIC_KEY_BYTES:
        raise ValueError(f"Invalid ML-DSA public key size: {len(mldsa_public_key)}")
    if len(mldsa_private_key) != config.DSA_PRIVATE_KEY_BYTES:
        raise ValueError(f"Invalid ML-DSA private key size: {len(mldsa_private_key)}")
    if len(mlkem_encap_key) != config.KEM_ENCAP_KEY_BYTES:
        raise ValueError(f"Invalid ML-KEM encap key size: {len(mlkem_encap_key)}")
    if len(mlkem_decap_key) != config.KEM_DECAP_KEY_BYTES:
        raise ValueError(f"Invalid ML-KEM decap key size: {len(mlkem_decap_key)}")

    mldsa_kp = MLDSAKeyPair(
        public_key=mldsa_public_key,
        _private_key=bytearray(mldsa_private_key),
    )
    mlkem_kp = KEMKeyPair(
        encap_key=mlkem_encap_key,
        _decap_key=bytearray(mlkem_decap_key),
    )

    return UserIdentity(
        user_id=user_id,
        display_name=display_name,
        mldsa_keypair=mldsa_kp,
        mlkem_keypair=mlkem_kp,
        certificate=certificate,
    )