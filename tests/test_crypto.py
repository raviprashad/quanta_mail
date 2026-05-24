"""
tests/test_crypto.py
─────────────────────
Unit tests for every cryptographic primitive.

Run with:
    pytest tests/test_crypto.py -v

Each test is self-contained — no server needed.
Tests verify correctness, rejection of bad inputs, and
secure deletion behaviour.
"""

import pytest
import time
from unittest.mock import patch


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — Utilities
# ══════════════════════════════════════════════════════════════════════════════

class TestUtils:
    """Tests for crypto/utils.py"""

    def test_secure_zero_wipes_bytearray(self):
        from crypto.utils import secure_zero
        secret = bytearray(b"super secret key material 1234!")
        original_len = len(secret)
        secure_zero(secret)
        assert all(b == 0 for b in secret), "secure_zero must set all bytes to 0"
        assert len(secret) == original_len, "secure_zero must not change length"

    def test_secure_zero_requires_bytearray(self):
        from crypto.utils import secure_zero
        with pytest.raises(TypeError):
            secure_zero(b"bytes are immutable")  # type: ignore

    def test_constant_time_compare_equal(self):
        from crypto.utils import constant_time_compare
        a = b"identical"
        b = b"identical"
        assert constant_time_compare(a, b) is True

    def test_constant_time_compare_different(self):
        from crypto.utils import constant_time_compare
        assert constant_time_compare(b"aaa", b"aab") is False
        assert constant_time_compare(b"short", b"longer string") is False

    def test_random_bytes_length(self):
        from crypto.utils import random_bytes
        for n in [16, 32, 64, 1184]:
            result = random_bytes(n)
            assert len(result) == n

    def test_random_bytes_not_all_zero(self):
        from crypto.utils import random_bytes
        result = random_bytes(32)
        assert any(b != 0 for b in result), "Random bytes must not be all zero"

    def test_random_bytes_each_call_different(self):
        from crypto.utils import random_bytes
        a = random_bytes(32)
        b = random_bytes(32)
        assert a != b, "Two random_bytes calls must produce different output"

    def test_random_bytes_rejects_zero_length(self):
        from crypto.utils import random_bytes
        with pytest.raises(ValueError):
            random_bytes(0)

    def test_sha3_256_deterministic(self):
        from crypto.utils import sha3_256
        assert sha3_256(b"hello") == sha3_256(b"hello")
        assert sha3_256(b"hello") != sha3_256(b"Hello")
        assert len(sha3_256(b"test")) == 32

    def test_sha3_512_length(self):
        from crypto.utils import sha3_512
        assert len(sha3_512(b"test")) == 64

    def test_transcript_hasher_order_matters(self):
        from crypto.utils import TranscriptHasher
        t1 = TranscriptHasher()
        t1.add(b"first")
        t1.add(b"second")

        t2 = TranscriptHasher()
        t2.add(b"second")
        t2.add(b"first")

        assert t1.digest() != t2.digest(), \
            "Transcript order must affect digest"

    def test_transcript_hasher_length_separation(self):
        """("ab","c") must differ from ("a","bc")"""
        from crypto.utils import TranscriptHasher
        t1 = TranscriptHasher()
        t1.add(b"ab")
        t1.add(b"c")

        t2 = TranscriptHasher()
        t2.add(b"a")
        t2.add(b"bc")

        assert t1.digest() != t2.digest()

    def test_public_key_fingerprint_format(self):
        from crypto.utils import public_key_fingerprint
        fp = public_key_fingerprint(b"\x00" * 64)
        assert ":" in fp
        parts = fp.split(":")
        assert len(parts) == 16, "Fingerprint should be 16 hex pairs"
        assert all(len(p) == 2 for p in parts)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — ML-KEM (Key Encapsulation)
# ══════════════════════════════════════════════════════════════════════════════

class TestMLKEM:
    """Tests for crypto/kem.py — FIPS 203 ML-KEM-768"""

    def test_keypair_sizes(self):
        from crypto.kem import generate_keypair
        import config
        kp = generate_keypair()
        assert len(kp.encap_key) == config.KEM_ENCAP_KEY_BYTES,   \
            f"encap_key must be {config.KEM_ENCAP_KEY_BYTES} bytes"
        assert len(kp.decap_key) == config.KEM_DECAP_KEY_BYTES,    \
            f"decap_key must be {config.KEM_DECAP_KEY_BYTES} bytes"
        kp.destroy()

    def test_each_keypair_is_unique(self):
        from crypto.kem import generate_keypair
        kp1 = generate_keypair()
        kp2 = generate_keypair()
        assert kp1.encap_key != kp2.encap_key, "Each keypair must be unique"
        kp1.destroy(); kp2.destroy()

    def test_encapsulation_output_sizes(self):
        from crypto.kem import generate_keypair, encapsulate
        import config
        kp = generate_keypair()
        result = encapsulate(kp.encap_key)
        assert len(result.ciphertext)    == config.KEM_CIPHERTEXT_BYTES
        assert len(result.shared_secret) == config.KEM_SHARED_SECRET_BYTES
        result.destroy(); kp.destroy()

    def test_full_kem_round_trip(self):
        """Alice generates keys, Bob encapsulates, Alice decapsulates → same secret."""
        from crypto.kem import generate_keypair, encapsulate, decapsulate

        # Alice
        alice_kp = generate_keypair()

        # Bob
        bob_result = encapsulate(alice_kp.encap_key)

        # Alice
        alice_secret = decapsulate(alice_kp.decap_key, bob_result.ciphertext)

        assert bytes(alice_secret) == bytes(bob_result.shared_secret), \
            "Decapsulated secret must match encapsulated secret"

        from crypto.utils import secure_zero
        secure_zero(alice_secret)
        bob_result.destroy()
        alice_kp.destroy()

    def test_different_keypairs_give_different_secrets(self):
        from crypto.kem import generate_keypair, encapsulate, decapsulate
        from crypto.utils import secure_zero

        kp1 = generate_keypair()
        kp2 = generate_keypair()

        result1 = encapsulate(kp1.encap_key)
        result2 = encapsulate(kp2.encap_key)

        secret1 = decapsulate(kp1.decap_key, result1.ciphertext)
        secret2 = decapsulate(kp2.decap_key, result2.ciphertext)

        assert bytes(secret1) != bytes(secret2)

        secure_zero(secret1); secure_zero(secret2)
        result1.destroy(); result2.destroy()
        kp1.destroy(); kp2.destroy()

    def test_wrong_decap_key_implicit_rejection(self):
        """
        FIPS 203 §6.3: decapsulating with the wrong key must NOT raise.
        It returns a pseudorandom value (implicit rejection).
        The session GCM tag will then fail on the first message.
        """
        from crypto.kem import generate_keypair, encapsulate, decapsulate
        from crypto.utils import secure_zero

        alice_kp = generate_keypair()
        eve_kp   = generate_keypair()   # wrong key

        result = encapsulate(alice_kp.encap_key)

        # Decapsulate with Eve's key — must not raise
        wrong_secret = decapsulate(eve_kp.decap_key, result.ciphertext)

        # The secret must differ from the correct one
        correct_secret = decapsulate(alice_kp.decap_key, result.ciphertext)
        assert bytes(wrong_secret) != bytes(correct_secret), \
            "Wrong decap key must produce different (random-looking) output"

        secure_zero(wrong_secret); secure_zero(correct_secret)
        result.destroy(); alice_kp.destroy(); eve_kp.destroy()

    def test_encap_key_type_check_rejects_wrong_size(self):
        from crypto.kem import encapsulate
        with pytest.raises(ValueError, match="length"):
            encapsulate(b"\x00" * 100)  # wrong size

    def test_destroy_zeros_decap_key(self):
        from crypto.kem import generate_keypair
        kp = generate_keypair()
        kp.destroy()
        assert all(b == 0 for b in kp._decap_key), \
            "destroy() must zero the private decap key"


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — ML-DSA (Digital Signatures)
# ══════════════════════════════════════════════════════════════════════════════

class TestMLDSA:
    """Tests for crypto/signing.py — FIPS 204 ML-DSA-65"""

    def test_keypair_sizes(self):
        from crypto.signing import mldsa_generate_keypair
        import config
        kp = mldsa_generate_keypair()
        assert len(kp.public_key)  == config.DSA_PUBLIC_KEY_BYTES
        assert len(kp.private_key) == config.DSA_PRIVATE_KEY_BYTES
        kp.destroy()

    def test_sign_and_verify(self):
        from crypto.signing import mldsa_generate_keypair, mldsa_sign, mldsa_verify
        kp  = mldsa_generate_keypair()
        msg = b"Hello, post-quantum world!"
        ctx = b"test-context"
        sig = mldsa_sign(kp.private_key, msg, ctx)
        assert mldsa_verify(kp.public_key, msg, sig, ctx)
        kp.destroy()

    def test_wrong_message_fails_verification(self):
        from crypto.signing import mldsa_generate_keypair, mldsa_sign, mldsa_verify
        kp  = mldsa_generate_keypair()
        sig = mldsa_sign(kp.private_key, b"original message", b"ctx")
        assert not mldsa_verify(kp.public_key, b"tampered message", sig, b"ctx")
        kp.destroy()

    def test_wrong_context_fails_verification(self):
        from crypto.signing import mldsa_generate_keypair, mldsa_sign, mldsa_verify
        kp  = mldsa_generate_keypair()
        sig = mldsa_sign(kp.private_key, b"message", b"context-a")
        assert not mldsa_verify(kp.public_key, b"message", sig, b"context-b"), \
            "Signature with context-a must not verify under context-b"
        kp.destroy()

    def test_wrong_public_key_fails_verification(self):
        from crypto.signing import mldsa_generate_keypair, mldsa_sign, mldsa_verify
        kp1 = mldsa_generate_keypair()
        kp2 = mldsa_generate_keypair()
        sig = mldsa_sign(kp1.private_key, b"message", b"ctx")
        assert not mldsa_verify(kp2.public_key, b"message", sig, b"ctx")
        kp1.destroy(); kp2.destroy()

    def test_tampered_signature_fails(self):
        from crypto.signing import mldsa_generate_keypair, mldsa_sign, mldsa_verify
        kp  = mldsa_generate_keypair()
        sig = bytearray(mldsa_sign(kp.private_key, b"message", b"ctx"))
        sig[42] ^= 0xFF   # flip bits
        assert not mldsa_verify(kp.public_key, b"message", bytes(sig), b"ctx")
        kp.destroy()

    def test_signature_size(self):
        from crypto.signing import mldsa_generate_keypair, mldsa_sign
        import config
        kp  = mldsa_generate_keypair()
        sig = mldsa_sign(kp.private_key, b"test", b"ctx")
        assert len(sig) == config.DSA_SIGNATURE_BYTES
        kp.destroy()

    def test_context_too_long_raises(self):
        from crypto.signing import mldsa_generate_keypair, mldsa_sign
        kp = mldsa_generate_keypair()
        with pytest.raises(ValueError, match="255"):
            mldsa_sign(kp.private_key, b"message", b"x" * 256)
        kp.destroy()

    def test_verify_wrong_sig_length_returns_false(self):
        from crypto.signing import mldsa_generate_keypair, mldsa_verify
        kp = mldsa_generate_keypair()
        assert mldsa_verify(kp.public_key, b"msg", b"\x00" * 10, b"ctx") is False
        kp.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — SLH-DSA (CA Signatures)
# ══════════════════════════════════════════════════════════════════════════════

class TestSLHDSA:
    """Tests for crypto/signing.py — FIPS 205 SLH-DSA proxy"""

    def test_keypair_sizes(self):
        from crypto.signing import slhdsa_generate_keypair
        import config
        kp = slhdsa_generate_keypair()
        assert len(kp.public_key)  == config.SLHDSA_PUBLIC_KEY_BYTES
        assert len(kp.private_key) == config.SLHDSA_PRIVATE_KEY_BYTES
        kp.destroy()

    def test_sign_and_verify(self):
        from crypto.signing import slhdsa_generate_keypair, slhdsa_sign, slhdsa_verify
        kp  = slhdsa_generate_keypair()
        msg = b"CA certificate signing"
        sig = slhdsa_sign(kp.private_key, msg)
        assert slhdsa_verify(kp.public_key, msg, sig)
        kp.destroy()

    def test_tampered_message_fails(self):
        from crypto.signing import slhdsa_generate_keypair, slhdsa_sign, slhdsa_verify
        kp  = slhdsa_generate_keypair()
        sig = slhdsa_sign(kp.private_key, b"original")
        assert not slhdsa_verify(kp.public_key, b"modified", sig)
        kp.destroy()

    def test_wrong_public_key_fails(self):
        from crypto.signing import slhdsa_generate_keypair, slhdsa_sign, slhdsa_verify
        kp1 = slhdsa_generate_keypair()
        kp2 = slhdsa_generate_keypair()
        sig = slhdsa_sign(kp1.private_key, b"message")
        assert not slhdsa_verify(kp2.public_key, b"message", sig)
        kp1.destroy(); kp2.destroy()

    def test_signature_size(self):
        from crypto.signing import slhdsa_generate_keypair, slhdsa_sign
        import config
        kp  = slhdsa_generate_keypair()
        sig = slhdsa_sign(kp.private_key, b"test")
        assert len(sig) == config.SLHDSA_SIGNATURE_BYTES
        kp.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — KDF
# ══════════════════════════════════════════════════════════════════════════════

class TestKDF:
    """Tests for crypto/kdf.py"""

    def test_session_keys_correct_sizes(self):
        from crypto.kdf import derive_session_keys
        import config
        secret    = b"\xAB" * 32
        transcript = b"\x00" * 64
        keys = derive_session_keys(secret, transcript, "client")
        assert len(keys.enc_key)  == config.SYMMETRIC_KEY_BYTES
        assert len(keys.mac_key)  == config.SYMMETRIC_KEY_BYTES
        assert len(keys.iv_seed)  == config.SYMMETRIC_KEY_BYTES
        keys.destroy()

    def test_client_and_server_keys_differ(self):
        """Same shared secret must produce different keys for client vs server."""
        from crypto.kdf import derive_session_keys
        secret     = b"\xCD" * 32
        transcript = b"\x11" * 64
        client_keys = derive_session_keys(secret, transcript, "client")
        server_keys = derive_session_keys(secret, transcript, "server")
        assert client_keys.enc_key != server_keys.enc_key, \
            "Client and server must have different enc keys"
        client_keys.destroy(); server_keys.destroy()

    def test_different_transcripts_give_different_keys(self):
        from crypto.kdf import derive_session_keys
        secret = b"\xEF" * 32
        keys1 = derive_session_keys(secret, b"\x00" * 64, "client")
        keys2 = derive_session_keys(secret, b"\x01" * 64, "client")
        assert keys1.enc_key != keys2.enc_key, \
            "Different transcripts must produce different keys"
        keys1.destroy(); keys2.destroy()

    def test_nonce_derivation_is_unique_per_seq(self):
        from crypto.kdf import derive_session_keys
        keys = derive_session_keys(b"\xAA" * 32, b"\xBB" * 64, "client")
        nonces = {keys.derive_nonce(i) for i in range(100)}
        assert len(nonces) == 100, "Each sequence number must produce a unique nonce"
        keys.destroy()

    def test_nonce_is_12_bytes(self):
        from crypto.kdf import derive_session_keys
        keys = derive_session_keys(b"\xCC" * 32, b"\xDD" * 64, "server")
        assert len(keys.derive_nonce(0)) == 12
        keys.destroy()

    def test_destroy_zeros_keys(self):
        from crypto.kdf import derive_session_keys
        keys = derive_session_keys(b"\x55" * 32, b"\x66" * 64, "client")
        keys.destroy()
        assert all(b == 0 for b in keys._enc_key)
        assert all(b == 0 for b in keys._mac_key)
        assert all(b == 0 for b in keys._iv_seed)

    def test_storage_key_derivation_deterministic(self):
        from crypto.kdf import derive_storage_key
        salt = b"\x12" * 32
        k1 = derive_storage_key("password123", salt)
        k2 = derive_storage_key("password123", salt)
        assert bytes(k1) == bytes(k2), \
            "Same password+salt must always give same storage key"
        from crypto.utils import secure_zero
        secure_zero(k1); secure_zero(k2)

    def test_storage_key_different_passwords_differ(self):
        from crypto.kdf import derive_storage_key
        from crypto.utils import secure_zero
        salt = b"\x34" * 32
        k1 = derive_storage_key("password1", salt)
        k2 = derive_storage_key("password2", salt)
        assert bytes(k1) != bytes(k2)
        secure_zero(k1); secure_zero(k2)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 6 — AES-256-GCM
# ══════════════════════════════════════════════════════════════════════════════

class TestSymmetric:
    """Tests for crypto/symmetric.py"""

    def _make_key_and_nonce(self):
        from crypto.utils import random_bytes
        return random_bytes(32), random_bytes(12)

    def test_encrypt_decrypt_round_trip(self):
        from crypto.symmetric import encrypt_message, decrypt_message
        key, nonce = self._make_key_and_nonce()
        plaintext  = b"Secret message content here"
        aad        = b"authenticated header"

        enc = encrypt_message(key, plaintext, nonce, aad)
        dec = decrypt_message(key, enc, aad)

        assert dec == plaintext

    def test_ciphertext_differs_from_plaintext(self):
        from crypto.symmetric import encrypt_message
        key, nonce = self._make_key_and_nonce()
        plaintext  = b"A" * 64
        enc = encrypt_message(key, plaintext, nonce)
        assert enc.ciphertext != plaintext

    def test_tampered_ciphertext_fails(self):
        from crypto.symmetric import encrypt_message, decrypt_message, EncryptedMessage
        key, nonce = self._make_key_and_nonce()
        enc = encrypt_message(key, b"payload", nonce)

        tampered_ct = bytes([enc.ciphertext[0] ^ 0xFF]) + enc.ciphertext[1:]
        bad_enc = EncryptedMessage(tampered_ct, enc.tag, enc.nonce)

        with pytest.raises(ValueError, match="authentication"):
            decrypt_message(key, bad_enc)

    def test_tampered_tag_fails(self):
        from crypto.symmetric import encrypt_message, decrypt_message, EncryptedMessage
        key, nonce = self._make_key_and_nonce()
        enc = encrypt_message(key, b"payload", nonce)

        bad_tag = bytes([enc.tag[0] ^ 0xFF]) + enc.tag[1:]
        bad_enc = EncryptedMessage(enc.ciphertext, bad_tag, enc.nonce)

        with pytest.raises(ValueError):
            decrypt_message(key, bad_enc)

    def test_tampered_aad_fails(self):
        from crypto.symmetric import encrypt_message, decrypt_message
        key, nonce = self._make_key_and_nonce()
        enc = encrypt_message(key, b"payload", nonce, b"original-header")
        with pytest.raises(ValueError):
            decrypt_message(key, enc, b"tampered-header")

    def test_wrong_key_fails(self):
        from crypto.symmetric import encrypt_message, decrypt_message
        from crypto.utils import random_bytes
        key1, nonce = self._make_key_and_nonce()
        key2 = random_bytes(32)
        enc = encrypt_message(key1, b"payload", nonce)
        with pytest.raises(ValueError):
            decrypt_message(key2, enc)

    def test_wrong_key_size_raises(self):
        from crypto.symmetric import encrypt_message
        from crypto.utils import random_bytes
        with pytest.raises(ValueError, match="AES"):
            encrypt_message(b"\x00" * 16, b"payload", random_bytes(12))

    def test_wrong_nonce_size_raises(self):
        from crypto.symmetric import encrypt_message
        from crypto.utils import random_bytes
        with pytest.raises(ValueError, match="nonce"):
            encrypt_message(random_bytes(32), b"payload", b"\x00" * 8)


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 7 — Certificates
# ══════════════════════════════════════════════════════════════════════════════

class TestCertificates:
    """Tests for identity/certificate.py"""

    @pytest.fixture
    def ca(self):
        from crypto.signing import slhdsa_generate_keypair
        from identity.certificate import CertificateAuthority
        kp = slhdsa_generate_keypair()
        return CertificateAuthority(kp)

    @pytest.fixture
    def user_keys(self):
        from crypto.signing import mldsa_generate_keypair
        from crypto.kem import generate_keypair
        mldsa_kp = mldsa_generate_keypair()
        mlkem_kp = generate_keypair()
        return mldsa_kp, mlkem_kp

    def test_issue_and_verify_certificate(self, ca, user_keys):
        mldsa_kp, mlkem_kp = user_keys
        cert = ca.issue_certificate(
            user_id="alice",
            display_name="Alice",
            mldsa_public_key=mldsa_kp.public_key,
            mlkem_encap_key=mlkem_kp.encap_key,
            validity_days=365,
        )
        # Must not raise
        ca.verify_certificate(cert)
        mldsa_kp.destroy(); mlkem_kp.destroy()

    def test_tampered_certificate_rejected(self, ca, user_keys):
        from identity.certificate import CertificateError
        mldsa_kp, mlkem_kp = user_keys
        cert = ca.issue_certificate(
            user_id="alice",
            display_name="Alice",
            mldsa_public_key=mldsa_kp.public_key,
            mlkem_encap_key=mlkem_kp.encap_key,
        )
        cert.display_name = "Eve"   # tamper with subject
        with pytest.raises(CertificateError):
            ca.verify_certificate(cert)
        mldsa_kp.destroy(); mlkem_kp.destroy()

    def test_wrong_ca_key_rejected(self, ca, user_keys):
        from crypto.signing import slhdsa_generate_keypair
        from identity.certificate import CertificateAuthority, CertificateError
        mldsa_kp, mlkem_kp = user_keys

        cert = ca.issue_certificate(
            user_id="alice",
            display_name="Alice",
            mldsa_public_key=mldsa_kp.public_key,
            mlkem_encap_key=mlkem_kp.encap_key,
        )

        evil_ca_kp = slhdsa_generate_keypair()
        evil_verifier = CertificateAuthority.verifier_only(evil_ca_kp.public_key)

        with pytest.raises(CertificateError):
            evil_verifier.verify_certificate(cert)

        evil_ca_kp.destroy()
        mldsa_kp.destroy(); mlkem_kp.destroy()

    def test_certificate_serialisation_round_trip(self, ca, user_keys):
        mldsa_kp, mlkem_kp = user_keys
        cert = ca.issue_certificate(
            user_id="alice",
            display_name="Alice Test",
            mldsa_public_key=mldsa_kp.public_key,
            mlkem_encap_key=mlkem_kp.encap_key,
        )
        serialised   = cert.to_bytes()
        deserialised = cert.from_bytes(serialised)

        assert deserialised.user_id      == cert.user_id
        assert deserialised.display_name == cert.display_name
        assert deserialised.mldsa_public_key == cert.mldsa_public_key
        assert deserialised.slhdsa_signature == cert.slhdsa_signature
        mldsa_kp.destroy(); mlkem_kp.destroy()

    def test_expired_certificate_rejected(self, ca, user_keys):
        from identity.certificate import CertificateError
        mldsa_kp, mlkem_kp = user_keys
        cert = ca.issue_certificate(
            user_id="alice",
            display_name="Alice",
            mldsa_public_key=mldsa_kp.public_key,
            mlkem_encap_key=mlkem_kp.encap_key,
            validity_days=365,
        )
        # Force expiry by manipulating timestamps
        cert.not_after = int(time.time()) - 1
        # Re-sign so the signature is valid but the cert is expired
        from crypto.signing import slhdsa_sign
        cert.slhdsa_signature = slhdsa_sign(ca._keypair.private_key, cert.tbs_bytes())

        with pytest.raises(CertificateError, match="validity"):
            ca.verify_certificate(cert)
        mldsa_kp.destroy(); mlkem_kp.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 8 — Key Storage
# ══════════════════════════════════════════════════════════════════════════════

class TestKeyStorage:
    """Tests for identity/storage.py"""

    def test_save_and_load_identity(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.KEY_STORAGE_DIR", str(tmp_path))

        from identity.keypair import generate_user_identity
        from identity.storage import save_identity, load_identity

        identity = generate_user_identity("testuser", "Test User")
        save_identity(identity, password="correct-password")

        loaded = load_identity("testuser", password="correct-password")

        assert loaded.user_id      == identity.user_id
        assert loaded.display_name == identity.display_name
        assert loaded.mldsa_public_key == identity.mldsa_public_key
        assert loaded.mlkem_encap_key  == identity.mlkem_encap_key
        assert loaded.mldsa_private_key == identity.mldsa_private_key

        identity.destroy(); loaded.destroy()

    def test_wrong_password_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.KEY_STORAGE_DIR", str(tmp_path))

        from identity.keypair import generate_user_identity
        from identity.storage import save_identity, load_identity

        identity = generate_user_identity("testuser2", "Test")
        save_identity(identity, password="correct")
        identity.destroy()

        with pytest.raises(ValueError, match="Decryption failed"):
            load_identity("testuser2", password="wrong")

    def test_missing_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("config.KEY_STORAGE_DIR", str(tmp_path))
        from identity.storage import load_identity
        with pytest.raises(FileNotFoundError):
            load_identity("nonexistent", "password")


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 9 — Full Handshake Protocol
# ══════════════════════════════════════════════════════════════════════════════

class TestHandshake:
    """
    Tests for protocol/handshake.py

    These tests exercise the full handshake without a network:
    client and server exchange bytes directly in memory.
    """

    @pytest.fixture
    def ca_and_verifier(self):
        from crypto.signing import slhdsa_generate_keypair
        from identity.certificate import CertificateAuthority
        kp = slhdsa_generate_keypair()
        ca = CertificateAuthority(kp)
        verifier = CertificateAuthority.verifier_only(kp.public_key)
        return ca, verifier

    @pytest.fixture
    def alice(self, ca_and_verifier):
        from identity.keypair import generate_user_identity
        ca, _ = ca_and_verifier
        identity = generate_user_identity("alice", "Alice")
        cert = ca.issue_certificate(
            user_id="alice", display_name="Alice",
            mldsa_public_key=identity.mldsa_public_key,
            mlkem_encap_key=identity.mlkem_encap_key,
        )
        identity.attach_certificate(cert)
        return identity

    @pytest.fixture
    def bob(self, ca_and_verifier):
        from identity.keypair import generate_user_identity
        ca, _ = ca_and_verifier
        identity = generate_user_identity("bob", "Bob")
        cert = ca.issue_certificate(
            user_id="bob", display_name="Bob",
            mldsa_public_key=identity.mldsa_public_key,
            mlkem_encap_key=identity.mlkem_encap_key,
        )
        identity.attach_certificate(cert)
        return identity

    def test_successful_handshake(self, alice, bob, ca_and_verifier):
        from protocol.handshake import ClientHandshake, ServerHandshake
        _, verifier = ca_and_verifier

        client_hs = ClientHandshake(alice, verifier)
        server_hs = ServerHandshake(bob, verifier)

        # Round 1: Alice → Bob
        client_hello = client_hs.build_client_hello()

        # Round 2: Bob → Alice
        server_hello = server_hs.process_client_hello(client_hello)

        # Round 3: Alice → Bob
        client_finished, client_session = client_hs.process_server_hello(server_hello)

        # Bob finalises
        server_session = server_hs.process_client_finished(client_finished)

        # Both sides must have derived identical session IDs
        assert client_session.session_id == server_session.session_id, \
            "Both sides must derive the same session_id"

        # Both sides must have derived the same session keys
        assert client_session.client_keys.enc_key == server_session.client_keys.enc_key
        assert client_session.server_keys.enc_key == server_session.server_keys.enc_key

        client_session.client_keys.destroy()
        client_session.server_keys.destroy()
        server_session.client_keys.destroy()
        server_session.server_keys.destroy()
        alice.destroy(); bob.destroy()

    def test_tampered_client_hello_rejected(self, alice, bob, ca_and_verifier):
        from protocol.handshake import ClientHandshake, ServerHandshake, HandshakeError
        _, verifier = ca_and_verifier

        client_hs = ClientHandshake(alice, verifier)
        server_hs = ServerHandshake(bob, verifier)

        client_hello = bytearray(client_hs.build_client_hello())
        client_hello[50] ^= 0xFF   # tamper

        with pytest.raises(HandshakeError):
            server_hs.process_client_hello(bytes(client_hello))

        alice.destroy(); bob.destroy()

    def test_replay_attack_rejected(self, alice, bob, ca_and_verifier):
        """A ClientHello with an old timestamp must be rejected."""
        from protocol.handshake import (
            ClientHandshake, ServerHandshake,
            ClientHello, HandshakeError,
        )
        from crypto.signing import mldsa_sign
        import config
        _, verifier = ca_and_verifier

        # Manually craft a ClientHello with an old timestamp
        from crypto.kem import generate_keypair
        kp = generate_keypair()
        old_hello = ClientHello(
            kem_encap_key=kp.encap_key,
            certificate=alice.certificate.to_bytes(),
            timestamp=int(time.time()) - 120,  # 2 minutes ago
        )
        old_hello.signature = mldsa_sign(
            alice.mldsa_private_key,
            old_hello.tbs_bytes(),
            config.DSA_CTX_HANDSHAKE,
        )
        server_hs = ServerHandshake(bob, verifier)

        with pytest.raises(HandshakeError, match="[Tt]imestamp"):
            server_hs.process_client_hello(old_hello.to_bytes())

        kp.destroy(); alice.destroy(); bob.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 10 — Session Messaging
# ══════════════════════════════════════════════════════════════════════════════

class TestSession:
    """Tests for protocol/session.py"""

    @pytest.fixture
    def established_sessions(self):
        """Run a complete handshake and return (client_session, server_session)."""
        from crypto.signing import slhdsa_generate_keypair
        from identity.certificate import CertificateAuthority
        from identity.keypair import generate_user_identity
        from protocol.handshake import ClientHandshake, ServerHandshake
        from protocol.session import Session, Role

        ca_kp    = slhdsa_generate_keypair()
        ca       = CertificateAuthority(ca_kp)
        verifier = CertificateAuthority.verifier_only(ca_kp.public_key)

        alice = generate_user_identity("alice", "Alice")
        bob   = generate_user_identity("bob", "Bob")

        for identity in [alice, bob]:
            cert = ca.issue_certificate(
                user_id=identity.user_id,
                display_name=identity.display_name,
                mldsa_public_key=identity.mldsa_public_key,
                mlkem_encap_key=identity.mlkem_encap_key,
            )
            identity.attach_certificate(cert)

        client_hs = ClientHandshake(alice, verifier)
        server_hs = ServerHandshake(bob, verifier)

        client_hello           = client_hs.build_client_hello()
        server_hello           = server_hs.process_client_hello(client_hello)
        client_finished, c_est = client_hs.process_server_hello(server_hello)
        s_est                  = server_hs.process_client_finished(client_finished)

        client_session = Session(c_est, alice, c_est.peer_certificate, Role.CLIENT)
        server_session = Session(s_est, bob,  s_est.peer_certificate, Role.SERVER)

        yield client_session, server_session, alice, bob

        client_session.close()
        server_session.close()
        alice.destroy()
        bob.destroy()

    def test_encrypt_decrypt_round_trip(self, established_sessions):
        client_session, server_session, alice, bob = established_sessions

        payload = "Hello Bob, this is Alice!".encode()
        envelope_bytes = client_session.encrypt(payload, "bob", "text")

        msg = server_session.decrypt(envelope_bytes)

        assert msg.payload  == payload
        assert msg.verified is True
        assert msg.header.sender_id    == "alice"
        assert msg.header.recipient_id == "bob"

    def test_multiple_messages_in_sequence(self, established_sessions):
        client_session, server_session, alice, bob = established_sessions

        for i in range(5):
            text    = f"Message {i}".encode()
            env     = client_session.encrypt(text, "bob")
            decoded = server_session.decrypt(env)
            assert decoded.payload == text

    def test_tampered_ciphertext_rejected(self, established_sessions):
        from protocol.session import SessionError
        client_session, server_session, alice, bob = established_sessions

        envelope_bytes = bytearray(client_session.encrypt(b"secret", "bob"))
        envelope_bytes[100] ^= 0xFF   # tamper deep in the ciphertext

        with pytest.raises(SessionError):
            server_session.decrypt(bytes(envelope_bytes))

    def test_tampered_header_rejected(self, established_sessions):
        """Tampering with the plaintext header must fail GCM tag check."""
        from protocol.session import SessionError
        import msgpack
        client_session, server_session, alice, bob = established_sessions

        envelope_bytes = client_session.encrypt(b"hello", "bob")
        envelope = __import__("protocol.session", fromlist=["EncryptedEnvelope"]
                              ).EncryptedEnvelope.from_bytes(envelope_bytes)

        # Tamper with the header bytes
        header_dict = msgpack.unpackb(envelope.header, raw=False)
        header_dict["recipient_id"] = "eve"
        envelope.header = msgpack.packb(header_dict, use_bin_type=True)

        with pytest.raises(SessionError):
            server_session.decrypt(envelope.to_bytes())

    def test_replay_detected(self, established_sessions):
        """Re-submitting an already-processed message must be rejected."""
        from protocol.session import SessionError
        client_session, server_session, alice, bob = established_sessions

        env = client_session.encrypt(b"once", "bob")
        server_session.decrypt(env)   # first time: OK

        with pytest.raises(SessionError, match="[Rr]eplay"):
            server_session.decrypt(env)   # second time: rejected

    def test_close_session_destroys_keys(self, established_sessions):
        from protocol.session import SessionError
        client_session, server_session, alice, bob = established_sessions

        client_session.close()

        with pytest.raises(SessionError, match="closed"):
            client_session.encrypt(b"after close", "bob")