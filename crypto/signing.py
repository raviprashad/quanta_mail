"""
crypto/signing.py
─────────────────
Digital signature wrappers.

  ML-DSA-65   (FIPS 204) — used for:
    • Per-message signatures
    • User identity signatures during handshake
    • Signing handshake transcripts

  SLH-DSA / SPHINCS+-SHAKE-192s  (FIPS 205) — used for:
    • Certificate Authority signing of user certificates
    • Long-lived trust anchors where performance is not critical

Why two signature schemes?
  ML-DSA is fast (~1ms per sign) making it suitable for frequent operations.
  SLH-DSA is slow (~300ms per sign) but its security rests only on hash
  function properties — no lattice assumptions.  Using it for CA signatures
  means the certificate infrastructure remains secure even if a future attack
  weakens the MLWE assumption.

FIPS 204 §3.4  — Hedged vs deterministic signing:
  We use the hedged variant by default (rnd generated fresh per signature).
  This adds randomness that mitigates side-channel attacks on the signing
  operation.  The same verification algorithm handles both variants.
"""

from __future__ import annotations
import oqs
from dataclasses import dataclass

import config
from crypto.utils import (
    secure_zero,
    public_key_fingerprint,
    random_bytes,
)


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class MLDSAKeyPair:
    """
    ML-DSA-65 identity key pair.

    public_key  – include in certificates and handshake messages (1952 bytes)
    _private_key – used only for signing; must call destroy() when retiring key
    """
    public_key: bytes
    _private_key: bytearray

    @property
    def private_key(self) -> bytes:
        return bytes(self._private_key)

    def destroy(self) -> None:
        """
        Zero the private key. Call when rotating or retiring this identity.
        FIPS 204 §3.6.3: intermediate and sensitive data must be destroyed
        when no longer needed.
        """
        secure_zero(self._private_key)

    @property
    def fingerprint(self) -> str:
        return public_key_fingerprint(self.public_key)


@dataclass
class SLHDSAKeyPair:
    """
    SLH-DSA (SPHINCS+-SHAKE-192s) key pair for Certificate Authorities.

    public_key  – 48 bytes, included in root CA record
    _private_key – 96 bytes, kept air-gapped and offline if possible
    """
    public_key: bytes
    _private_key: bytearray

    @property
    def private_key(self) -> bytes:
        return bytes(self._private_key)

    def destroy(self) -> None:
        secure_zero(self._private_key)

    @property
    def fingerprint(self) -> str:
        return public_key_fingerprint(self.public_key)


# ─── ML-DSA functions ─────────────────────────────────────────────────────────

def mldsa_generate_keypair() -> MLDSAKeyPair:
    """
    Generate a fresh ML-DSA-65 key pair.

    For user identity keys this is called once and the result stored
    in encrypted key storage.  The public key goes into the certificate.

    Returns:
        MLDSAKeyPair with public_key and private_key.
    """
    sig = oqs.Signature(config.DSA_ALGORITHM)
    public_key: bytes = sig.generate_keypair()
    private_key: bytes = sig.export_secret_key()

    _assert_size("mldsa_public_key",  public_key,  config.DSA_PUBLIC_KEY_BYTES)
    _assert_size("mldsa_private_key", private_key, config.DSA_PRIVATE_KEY_BYTES)

    return MLDSAKeyPair(
        public_key=public_key,
        _private_key=bytearray(private_key),
    )


def mldsa_sign(
    private_key: bytes,
    message: bytes,
    context: bytes = b"",
) -> bytes:
    """
    Sign a message with ML-DSA-65.

    FIPS 204 §5.2 context string:
      The context provides domain separation so a signature produced for
      one purpose (e.g. message signing) cannot be replayed as a signature
      for another purpose (e.g. certificate signing).  Always pass the
      appropriate DSA_CTX_* constant from config.py.

    FIPS 204 §3.4 hedged signing:
      liboqs generates fresh randomness internally for the hedged variant.
      We don't need to pass rnd explicitly — liboqs handles this.

    Args:
        private_key:  ML-DSA-65 private key bytes.
        message:      Arbitrary bytes to sign.
        context:      Domain separation context string (max 255 bytes).

    Returns:
        Signature bytes (3309 bytes for ML-DSA-65).

    Raises:
        ValueError  if context exceeds 255 bytes (FIPS 204 §5.2).
        RuntimeError if signature size is unexpected.
    """
    if len(context) > 255:
        raise ValueError(
            f"Context string must be ≤ 255 bytes (FIPS 204 §5.2), "
            f"got {len(context)}"
        )

    sig = oqs.Signature(config.DSA_ALGORITHM, secret_key=private_key)

    # liboqs ML-DSA.Sign handles hedged randomness internally.
    # We pass message and context using the context-aware sign method.
    signature: bytes = sig.sign_with_ctx_str(message, context)

    _assert_size("mldsa_signature", signature, config.DSA_SIGNATURE_BYTES)
    return signature


def mldsa_verify(
    public_key: bytes,
    message: bytes,
    signature: bytes,
    context: bytes = b"",
) -> bool:
    """
    Verify an ML-DSA-65 signature.

    FIPS 204 §3.6.2: if the signature length differs from the expected
    value, return False immediately (do not pass to the library).

    Args:
        public_key: ML-DSA-65 public key of the alleged signer.
        message:    The original message that was signed.
        signature:  The signature to verify.
        context:    Must match the context used at signing time.

    Returns:
        True if the signature is valid, False otherwise.
        Never raises an exception on invalid signatures — always returns False.
    """
    # FIPS 204 §3.6.2: length check before calling algorithm
    if len(signature) != config.DSA_SIGNATURE_BYTES:
        return False
    if len(public_key) != config.DSA_PUBLIC_KEY_BYTES:
        return False
    if len(context) > 255:
        return False

    try:
        sig = oqs.Signature(config.DSA_ALGORITHM)
        return sig.verify_with_ctx_str(message, signature, context, public_key)
    except Exception:
        # Any exception from the library means invalid signature.
        return False


# ─── SLH-DSA functions ────────────────────────────────────────────────────────

def slhdsa_generate_keypair() -> SLHDSAKeyPair:
    """
    Generate an SLH-DSA (SPHINCS+-SHAKE-192s) key pair for a CA.

    This is called once when setting up a CA.  The private key should
    be stored in an HSM or air-gapped machine.

    FIPS 205 §3.1: PK.seed, SK.seed, and SK.prf must be generated by
    an approved RBG with at least 8n = 192 bits of security.
    liboqs handles this internally via the OS CSPRNG.

    Returns:
        SLHDSAKeyPair with public_key (48 bytes) and private_key (96 bytes).
    """
    sig = oqs.Signature(config.SLHDSA_ALGORITHM)
    public_key: bytes = sig.generate_keypair()
    private_key: bytes = sig.export_secret_key()

    _assert_size("slhdsa_public_key",  public_key,  config.SLHDSA_PUBLIC_KEY_BYTES)
    _assert_size("slhdsa_private_key", private_key, config.SLHDSA_PRIVATE_KEY_BYTES)

    return SLHDSAKeyPair(
        public_key=public_key,
        _private_key=bytearray(private_key),
    )


def slhdsa_sign(private_key: bytes, message: bytes) -> bytes:
    """
    Sign a message with SLH-DSA (SPHINCS+-SHAKE-192s).

    Used only for signing certificates.  Slow (~300ms) but appropriate
    here since certificates are signed infrequently.

    FIPS 205 §10.2.1: hedged variant is the default and is used here.
    The 'addrand' value is generated internally by liboqs.

    Returns:
        Signature bytes (16224 bytes for SPHINCS+-SHAKE-192s).
    """
    sig = oqs.Signature(config.SLHDSA_ALGORITHM, secret_key=private_key)
    signature: bytes = sig.sign(message)

    _assert_size("slhdsa_signature", signature, config.SLHDSA_SIGNATURE_BYTES)
    return signature


def slhdsa_verify(
    public_key: bytes,
    message: bytes,
    signature: bytes,
) -> bool:
    """
    Verify an SLH-DSA (SPHINCS+-SHAKE-192s) signature.

    Returns True if valid, False otherwise.  Never raises on bad sigs.
    """
    if len(signature) != config.SLHDSA_SIGNATURE_BYTES:
        return False
    if len(public_key) != config.SLHDSA_PUBLIC_KEY_BYTES:
        return False

    try:
        sig = oqs.Signature(config.SLHDSA_ALGORITHM)
        return sig.verify(message, signature, public_key)
    except Exception:
        return False


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _assert_size(name: str, data: bytes, expected: int) -> None:
    if len(data) != expected:
        raise RuntimeError(
            f"Internal error: {name} has unexpected size "
            f"(got {len(data)}, expected {expected}). "
            "Check your liboqs version matches config.py."
        )