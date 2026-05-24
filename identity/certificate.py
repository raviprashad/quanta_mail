"""
identity/certificate.py
────────────────────────
Post-quantum X.509-style certificates.

A certificate ties a user's identity (name / user_id) to their
public keys (ML-DSA-65 for signing, ML-KEM-768 for key encapsulation).
It is signed by a Certificate Authority using SLH-DSA.

Certificate structure:
  ┌─────────────────────────────────────────────┐
  │  subject:    user_id, display_name          │
  │  not_before: Unix timestamp                  │
  │  not_after:  Unix timestamp                  │
  │  mldsa_pk:   ML-DSA-65 public key (1952 B)  │
  │  mlkem_pk:   ML-KEM-768 encap key (1184 B)  │
  │  issuer_id:  fingerprint of CA public key   │
  ├─────────────────────────────────────────────┤
  │  slhdsa_sig: CA's SLH-DSA signature over    │
  │              all fields above (16224 B)      │
  └─────────────────────────────────────────────┘

We serialise to/from msgpack (binary, compact) rather than ASN.1/DER
to keep the code readable.  In production you would use proper X.509
with ML-DSA/SLH-DSA OIDs, but the cryptographic logic is identical.
"""

from __future__ import annotations
import time
import json
import hashlib
from dataclasses import dataclass, field
from typing import Optional

import msgpack

from crypto.signing import (
    mldsa_verify,
    slhdsa_sign,
    slhdsa_verify,
    SLHDSAKeyPair,
)
from crypto.utils import sha3_256, public_key_fingerprint
import config


# ─── Certificate data class ───────────────────────────────────────────────────

@dataclass
class Certificate:
    """
    Represents a single user's post-quantum certificate.

    All fields except slhdsa_signature form the "to-be-signed" blob.
    The signature covers exactly that blob.
    """
    # ── Identity fields ───────────────────────────────────────────────────────
    user_id:      str     # unique identifier (UUID or username)
    display_name: str     # human-readable name
    not_before:   int     # Unix timestamp: certificate validity start
    not_after:    int     # Unix timestamp: certificate validity end

    # ── Public keys ───────────────────────────────────────────────────────────
    mldsa_public_key: bytes   # ML-DSA-65  public key (for verifying signatures)
    mlkem_encap_key:  bytes   # ML-KEM-768 encapsulation key (for key exchange)

    # ── CA signature ──────────────────────────────────────────────────────────
    issuer_id:     str    # fingerprint of the CA's SLH-DSA public key
    slhdsa_signature: bytes = field(default=b"")  # filled in by CA

    # ── Computed property ─────────────────────────────────────────────────────
    @property
    def fingerprint(self) -> str:
        """SHA3-256 fingerprint of this certificate's ML-DSA public key."""
        return public_key_fingerprint(self.mldsa_public_key)

    @property
    def is_expired(self) -> bool:
        now = int(time.time())
        return now < self.not_before or now > self.not_after

    def tbs_bytes(self) -> bytes:
        """
        Return the To-Be-Signed portion as a canonical byte string.
        The CA signs exactly this blob; verifiers check exactly this blob.
        Order matters — never change this layout without a version bump.
        """
        tbs = {
            "user_id":         self.user_id,
            "display_name":    self.display_name,
            "not_before":      self.not_before,
            "not_after":       self.not_after,
            "mldsa_public_key": self.mldsa_public_key,
            "mlkem_encap_key":  self.mlkem_encap_key,
            "issuer_id":       self.issuer_id,
        }
        return msgpack.packb(tbs, use_bin_type=True)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_bytes(self) -> bytes:
        """Serialise the complete certificate (TBS + signature) to bytes."""
        data = {
            "user_id":          self.user_id,
            "display_name":     self.display_name,
            "not_before":       self.not_before,
            "not_after":        self.not_after,
            "mldsa_public_key": self.mldsa_public_key,
            "mlkem_encap_key":  self.mlkem_encap_key,
            "issuer_id":        self.issuer_id,
            "slhdsa_signature": self.slhdsa_signature,
        }
        return msgpack.packb(data, use_bin_type=True)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Certificate":
        """Deserialise a certificate from bytes."""
        d = msgpack.unpackb(data, raw=False)
        return cls(
            user_id=d["user_id"],
            display_name=d["display_name"],
            not_before=d["not_before"],
            not_after=d["not_after"],
            mldsa_public_key=bytes(d["mldsa_public_key"]),
            mlkem_encap_key=bytes(d["mlkem_encap_key"]),
            issuer_id=d["issuer_id"],
            slhdsa_signature=bytes(d["slhdsa_signature"]),
        )


# ─── Certificate Authority ────────────────────────────────────────────────────

class CertificateAuthority:
    """
    Issues and verifies post-quantum certificates.

    The CA holds an SLH-DSA key pair.  It signs the TBS portion of each
    user certificate.  Verifiers only need the CA's public key.

    In production the CA private key would live in an HSM.
    Here we keep it in memory for simplicity.

    Usage:
        # Set up CA once
        ca_keypair = slhdsa_generate_keypair()
        ca = CertificateAuthority(ca_keypair)

        # Issue a certificate
        cert = ca.issue_certificate(
            user_id="alice",
            display_name="Alice Example",
            mldsa_public_key=alice_mldsa_pk,
            mlkem_encap_key=alice_mlkem_ek,
            validity_days=365,
        )

        # Verify a certificate (receiver side — only needs ca_public_key)
        verifier = CertificateAuthority.verifier_only(ca_keypair.public_key)
        verifier.verify_certificate(cert)   # raises on failure
    """

    def __init__(self, keypair: SLHDSAKeyPair):
        self._keypair = keypair
        self.public_key = keypair.public_key
        self.fingerprint = public_key_fingerprint(keypair.public_key)

    @classmethod
    def verifier_only(cls, ca_public_key: bytes) -> "_CAVerifier":
        """
        Return a verifier that can check certificates but cannot issue them.
        Clients use this — they never need the CA's private key.
        """
        return _CAVerifier(ca_public_key)

    def issue_certificate(
        self,
        user_id: str,
        display_name: str,
        mldsa_public_key: bytes,
        mlkem_encap_key: bytes,
        validity_days: int = 365,
    ) -> Certificate:
        """
        Issue and sign a certificate for a user.

        Creates the TBS (to-be-signed) portion, signs it with SLH-DSA,
        and returns the complete certificate.

        Args:
            user_id:           Unique user identifier.
            display_name:      Human-readable name.
            mldsa_public_key:  User's ML-DSA-65 public key.
            mlkem_encap_key:   User's ML-KEM-768 encapsulation key.
            validity_days:     Certificate lifetime in days.

        Returns:
            Signed Certificate.
        """
        now = int(time.time())

        cert = Certificate(
            user_id=user_id,
            display_name=display_name,
            not_before=now,
            not_after=now + validity_days * 86400,
            mldsa_public_key=mldsa_public_key,
            mlkem_encap_key=mlkem_encap_key,
            issuer_id=self.fingerprint,
        )

        # Sign the TBS portion with SLH-DSA
        tbs = cert.tbs_bytes()
        cert.slhdsa_signature = slhdsa_sign(
            private_key=self._keypair.private_key,
            message=tbs,
        )

        return cert

    def verify_certificate(self, cert: Certificate) -> None:
        """
        Verify a certificate's signature and validity period.
        Raises CertificateError on any failure.
        """
        _verify_cert(cert, self.public_key)


class _CAVerifier:
    """Verifier-only view of a CA — no private key."""

    def __init__(self, ca_public_key: bytes):
        self.public_key = ca_public_key

    def verify_certificate(self, cert: Certificate) -> None:
        _verify_cert(cert, self.public_key)


def _verify_cert(cert: Certificate, ca_public_key: bytes) -> None:
    """
    Internal certificate verification logic.

    Checks:
      1. Signature is present and correct length.
      2. SLH-DSA signature over TBS is valid.
      3. Certificate is within its validity period.
      4. Public key sizes match expected values for the algorithms.

    Raises:
        CertificateError with a descriptive message on any failure.
    """
    # ── Check 1: signature present ────────────────────────────────────────────
    if not cert.slhdsa_signature:
        raise CertificateError("Certificate has no signature")

    # ── Check 2: SLH-DSA signature valid ─────────────────────────────────────
    tbs = cert.tbs_bytes()
    if not slhdsa_verify(ca_public_key, tbs, cert.slhdsa_signature):
        raise CertificateError(
            f"Certificate signature invalid for user '{cert.user_id}'"
        )

    # ── Check 3: validity period ──────────────────────────────────────────────
    if cert.is_expired:
        now = int(time.time())
        raise CertificateError(
            f"Certificate for '{cert.user_id}' is outside validity period "
            f"(not_before={cert.not_before}, not_after={cert.not_after}, now={now})"
        )

    # ── Check 4: key sizes ────────────────────────────────────────────────────
    if len(cert.mldsa_public_key) != config.DSA_PUBLIC_KEY_BYTES:
        raise CertificateError(
            f"Certificate ML-DSA public key has wrong size: "
            f"{len(cert.mldsa_public_key)} (expected {config.DSA_PUBLIC_KEY_BYTES})"
        )
    if len(cert.mlkem_encap_key) != config.KEM_ENCAP_KEY_BYTES:
        raise CertificateError(
            f"Certificate ML-KEM encap key has wrong size: "
            f"{len(cert.mlkem_encap_key)} (expected {config.KEM_ENCAP_KEY_BYTES})"
        )


class CertificateError(Exception):
    """Raised when certificate verification fails."""
    pass