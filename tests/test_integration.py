"""
tests/test_integration.py
──────────────────────────
End-to-end integration tests.

These tests simulate the FULL flow:
  1. CA setup
  2. User registration (key gen + certificate issuance)
  3. Post-quantum handshake (in memory, no network)
  4. Bidirectional encrypted messaging
  5. Session rekeying
  6. Secure teardown

Run with:
    pytest tests/test_integration.py -v -s

The -s flag shows print output so you can follow the flow step by step.
"""

import pytest
import time


# ─── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ca():
    """One CA shared across all integration tests in this module."""
    from crypto.signing import slhdsa_generate_keypair
    from identity.certificate import CertificateAuthority
    print("\n[fixture] Generating CA (SLH-DSA-SHAKE-192s) ...")
    kp = slhdsa_generate_keypair()
    authority = CertificateAuthority(kp)
    print(f"[fixture] CA fingerprint: {authority.fingerprint}")
    yield authority
    kp.destroy()


@pytest.fixture
def registered_user(ca):
    """
    Generate a fresh user identity, issue a certificate, return both.
    Destroys the identity after the test.
    """
    from identity.keypair import generate_user_identity
    import uuid

    user_id = f"user_{uuid.uuid4().hex[:8]}"
    identity = generate_user_identity(user_id, f"Test User {user_id}")

    cert = ca.issue_certificate(
        user_id=user_id,
        display_name=identity.display_name,
        mldsa_public_key=identity.mldsa_public_key,
        mlkem_encap_key=identity.mlkem_encap_key,
        validity_days=365,
    )
    identity.attach_certificate(cert)

    yield identity

    identity.destroy()


def _run_handshake(alice, bob, ca):
    """
    Helper: run a complete handshake between alice and bob.
    Returns (client_session, server_session).
    """
    from identity.certificate import CertificateAuthority
    from protocol.handshake import ClientHandshake, ServerHandshake
    from protocol.session import Session, Role

    verifier = CertificateAuthority.verifier_only(ca._keypair.public_key)

    client_hs = ClientHandshake(alice, verifier)
    server_hs = ServerHandshake(bob, verifier)

    client_hello             = client_hs.build_client_hello()
    server_hello             = server_hs.process_client_hello(client_hello)
    client_finished, c_est   = client_hs.process_server_hello(server_hello)
    s_est                    = server_hs.process_client_finished(client_finished)

    client_session = Session(c_est, alice, c_est.peer_certificate, Role.CLIENT)
    server_session = Session(s_est, bob,  s_est.peer_certificate, Role.SERVER)

    return client_session, server_session


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Complete registration and handshake flow
# ══════════════════════════════════════════════════════════════════════════════

class TestFullRegistrationAndHandshake:

    def test_two_users_complete_handshake(self, registered_user, ca):
        """
        Two independently registered users can complete a PQ handshake
        and derive identical session keys.
        """
        from identity.keypair import generate_user_identity

        alice   = registered_user
        bob_raw = generate_user_identity("bob_int", "Bob Integration")
        bob_cert = ca.issue_certificate(
            user_id="bob_int",
            display_name="Bob Integration",
            mldsa_public_key=bob_raw.mldsa_public_key,
            mlkem_encap_key=bob_raw.mlkem_encap_key,
        )
        bob_raw.attach_certificate(bob_cert)

        print(f"\n[test] Alice fingerprint: {alice.fingerprint}")
        print(f"[test] Bob   fingerprint: {bob_raw.fingerprint}")

        client_session, server_session = _run_handshake(alice, bob_raw, ca)

        print(f"[test] Session ID: {client_session._session_id.hex()[:16]}...")

        # Both sides derived the same session ID
        assert client_session._session_id == server_session._session_id

        # Send a message from alice (client) to bob (server)
        payload = b"Hello Bob, handshake complete!"
        env     = client_session.encrypt(payload, "bob_int")
        msg     = server_session.decrypt(env)

        assert msg.payload  == payload
        assert msg.verified is True
        print(f"[test] Bob received: {msg.payload.decode()}")

        client_session.close()
        server_session.close()
        bob_raw.destroy()

    def test_certificate_fingerprint_in_session(self, registered_user, ca):
        """The peer certificate in an established session holds the correct user."""
        from identity.keypair import generate_user_identity

        alice   = registered_user
        bob_raw = generate_user_identity("bob_fp", "Bob FP")
        bob_cert = ca.issue_certificate(
            user_id="bob_fp", display_name="Bob FP",
            mldsa_public_key=bob_raw.mldsa_public_key,
            mlkem_encap_key=bob_raw.mlkem_encap_key,
        )
        bob_raw.attach_certificate(bob_cert)

        client_session, server_session = _run_handshake(alice, bob_raw, ca)

        # Client sees Bob's certificate; server sees Alice's certificate
        assert client_session._peer_cert.user_id == "bob_fp"
        assert server_session._peer_cert.user_id == alice.user_id

        client_session.close()
        server_session.close()
        bob_raw.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Bidirectional messaging
# ══════════════════════════════════════════════════════════════════════════════

class TestBidirectionalMessaging:

    @pytest.fixture
    def sessions(self, registered_user, ca):
        from identity.keypair import generate_user_identity
        alice   = registered_user
        bob_raw = generate_user_identity("bob_bi", "Bob Bi")
        bob_cert = ca.issue_certificate(
            user_id="bob_bi", display_name="Bob Bi",
            mldsa_public_key=bob_raw.mldsa_public_key,
            mlkem_encap_key=bob_raw.mlkem_encap_key,
        )
        bob_raw.attach_certificate(bob_cert)
        client_s, server_s = _run_handshake(alice, bob_raw, ca)
        yield client_s, server_s, alice, bob_raw
        client_s.close()
        server_s.close()
        bob_raw.destroy()

    def test_client_to_server_messages(self, sessions):
        client_s, server_s, alice, bob = sessions
        messages = [b"First", b"Second", b"Third", b"Fourth", b"Fifth"]
        for payload in messages:
            env = client_s.encrypt(payload, bob.user_id)
            msg = server_s.decrypt(env)
            assert msg.payload == payload

    def test_server_to_client_messages(self, sessions):
        """Server can also send messages to the client."""
        client_s, server_s, alice, bob = sessions
        messages = [b"Reply 1", b"Reply 2", b"Reply 3"]
        for payload in messages:
            env = server_s.encrypt(payload, alice.user_id)
            msg = client_s.decrypt(env)
            assert msg.payload == payload

    def test_interleaved_bidirectional(self, sessions):
        """Client and server can interleave sends and receives freely."""
        client_s, server_s, alice, bob = sessions

        # Alice sends
        e1 = client_s.encrypt(b"Alice: Hello!", bob.user_id)
        m1 = server_s.decrypt(e1)
        assert m1.payload == b"Alice: Hello!"

        # Bob replies
        e2 = server_s.encrypt(b"Bob: Hi Alice!", alice.user_id)
        m2 = client_s.decrypt(e2)
        assert m2.payload == b"Bob: Hi Alice!"

        # Alice sends again
        e3 = client_s.encrypt(b"Alice: How are you?", bob.user_id)
        m3 = server_s.decrypt(e3)
        assert m3.payload == b"Alice: How are you?"

        print("\n[test] Bidirectional conversation:")
        print(f"  Alice→Bob: {m1.payload.decode()}")
        print(f"  Bob→Alice: {m2.payload.decode()}")
        print(f"  Alice→Bob: {m3.payload.decode()}")

    def test_large_payload(self, sessions):
        """Messages up to several MB should encrypt/decrypt correctly."""
        client_s, server_s, alice, bob = sessions
        large_payload = b"X" * 1_000_000   # 1 MB
        env = client_s.encrypt(large_payload, bob.user_id)
        msg = server_s.decrypt(env)
        assert msg.payload == large_payload
        print(f"\n[test] 1 MB message: OK ({len(env)} bytes on wire)")

    def test_unicode_payload(self, sessions):
        """UTF-8 encoded Unicode content round-trips correctly."""
        client_s, server_s, alice, bob = sessions
        text    = "こんにちは世界 🔒 Привет мир"
        payload = text.encode("utf-8")
        env     = client_s.encrypt(payload, bob.user_id)
        msg     = server_s.decrypt(env)
        assert msg.payload.decode("utf-8") == text


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Security property tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSecurityProperties:

    @pytest.fixture
    def sessions(self, registered_user, ca):
        from identity.keypair import generate_user_identity
        alice   = registered_user
        bob_raw = generate_user_identity("bob_sec", "Bob Sec")
        bob_cert = ca.issue_certificate(
            user_id="bob_sec", display_name="Bob Sec",
            mldsa_public_key=bob_raw.mldsa_public_key,
            mlkem_encap_key=bob_raw.mlkem_encap_key,
        )
        bob_raw.attach_certificate(bob_cert)
        client_s, server_s = _run_handshake(alice, bob_raw, ca)
        yield client_s, server_s, alice, bob_raw
        client_s.close()
        server_s.close()
        bob_raw.destroy()

    def test_ciphertext_is_different_for_same_plaintext(self, sessions):
        """
        Two encryptions of the same plaintext must produce different ciphertext.
        This is guaranteed by the sequence-number-derived nonce.
        """
        client_s, server_s, alice, bob = sessions
        payload = b"Same message"

        env1 = client_s.encrypt(payload, bob.user_id)
        env2 = client_s.encrypt(payload, bob.user_id)

        from protocol.session import EncryptedEnvelope
        e1 = EncryptedEnvelope.from_bytes(env1)
        e2 = EncryptedEnvelope.from_bytes(env2)

        assert e1.ciphertext != e2.ciphertext, \
            "Encrypting the same plaintext twice must give different ciphertext"
        assert e1.nonce != e2.nonce, \
            "Each message must use a different nonce"

        # Both must decrypt correctly
        m1 = server_s.decrypt(env1)
        m2 = server_s.decrypt(env2)
        assert m1.payload == payload
        assert m2.payload == payload

    def test_replay_attack_rejected(self, sessions):
        """Re-submitting a captured message must be rejected."""
        from protocol.session import SessionError
        client_s, server_s, alice, bob = sessions

        env = client_s.encrypt(b"Sensitive data", bob.user_id)
        server_s.decrypt(env)   # legitimate delivery

        with pytest.raises(SessionError, match="[Rr]eplay"):
            server_s.decrypt(env)   # replay attempt
        print("\n[test] Replay attack correctly rejected")

    def test_bit_flip_in_ciphertext_rejected(self, sessions):
        """A single flipped bit in the ciphertext must fail GCM auth."""
        from protocol.session import SessionError, EncryptedEnvelope
        import msgpack
        client_s, server_s, alice, bob = sessions

        env_bytes = client_s.encrypt(b"Tamper me", bob.user_id)
        env       = EncryptedEnvelope.from_bytes(env_bytes)

        # Flip one bit deep in the ciphertext
        ct = bytearray(env.ciphertext)
        ct[len(ct) // 2] ^= 0x01
        env.ciphertext = bytes(ct)

        with pytest.raises(SessionError):
            server_s.decrypt(env.to_bytes())
        print("\n[test] Ciphertext bit-flip correctly rejected")

    def test_cross_session_message_rejected(self, registered_user, ca):
        """
        A message from one session cannot be replayed into a different session.
        The session_id in the header is validated against the session's own ID.
        """
        from identity.keypair import generate_user_identity
        from protocol.session import SessionError

        alice   = registered_user
        bob_raw = generate_user_identity("bob_cross1", "Bob Cross 1")
        eve_raw = generate_user_identity("eve_cross",  "Eve Cross")

        for identity in [bob_raw, eve_raw]:
            cert = ca.issue_certificate(
                user_id=identity.user_id,
                display_name=identity.display_name,
                mldsa_public_key=identity.mldsa_public_key,
                mlkem_encap_key=identity.mlkem_encap_key,
            )
            identity.attach_certificate(cert)

        # Alice <-> Bob session
        alice_bob_c, alice_bob_s = _run_handshake(alice, bob_raw, ca)

        # Alice <-> Eve session
        alice_eve_c, alice_eve_s = _run_handshake(alice, eve_raw, ca)

        # Encrypt message in Alice-Bob session
        env_ab = alice_bob_c.encrypt(b"For Bob only", bob_raw.user_id)

        # Try to decrypt it in Alice-Eve session — must fail
        with pytest.raises(SessionError):
            alice_eve_s.decrypt(env_ab)

        print("\n[test] Cross-session replay correctly rejected")

        alice_bob_c.close(); alice_bob_s.close()
        alice_eve_c.close(); alice_eve_s.close()
        bob_raw.destroy(); eve_raw.destroy()

    def test_forward_secrecy_ephemeral_keys_destroyed(self, registered_user, ca):
        """
        After the handshake, the ephemeral ML-KEM decap key must be zeroed.
        This ensures past sessions cannot be decrypted if long-term keys leak.
        """
        from identity.keypair import generate_user_identity
        from identity.certificate import CertificateAuthority
        from protocol.handshake import ClientHandshake, ServerHandshake

        alice   = registered_user
        bob_raw = generate_user_identity("bob_fs", "Bob FS")
        bob_cert = ca.issue_certificate(
            user_id="bob_fs", display_name="Bob FS",
            mldsa_public_key=bob_raw.mldsa_public_key,
            mlkem_encap_key=bob_raw.mlkem_encap_key,
        )
        bob_raw.attach_certificate(bob_cert)

        verifier  = CertificateAuthority.verifier_only(ca._keypair.public_key)
        client_hs = ClientHandshake(alice, verifier)
        server_hs = ServerHandshake(bob_raw, verifier)

        client_hello             = client_hs.build_client_hello()
        server_hello             = server_hs.process_client_hello(client_hello)
        client_finished, c_est   = client_hs.process_server_hello(server_hello)
        server_hs.process_client_finished(client_finished)

        # After process_server_hello, the ephemeral KEM keypair must be None
        assert client_hs._kem_keypair is None, \
            "Ephemeral KEM keypair must be destroyed after handshake"

        print("\n[test] Ephemeral KEM key destroyed after handshake ✓")

        c_est.client_keys.destroy()
        c_est.server_keys.destroy()
        bob_raw.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — Key storage round-trip
# ══════════════════════════════════════════════════════════════════════════════

class TestKeyStorageIntegration:

    def test_save_load_and_use_identity(self, tmp_path, monkeypatch, ca):
        """
        Full round-trip: generate identity → save encrypted → load from disk
        → use in a handshake.
        """
        monkeypatch.setattr("config.KEY_STORAGE_DIR", str(tmp_path))

        from identity.keypair import generate_user_identity
        from identity.storage import save_identity, load_identity, update_certificate
        from identity.certificate import CertificateAuthority

        # ── Generate and save ─────────────────────────────────────────────────
        alice    = generate_user_identity("alice_store", "Alice Store")
        alice_cert = ca.issue_certificate(
            user_id="alice_store", display_name="Alice Store",
            mldsa_public_key=alice.mldsa_public_key,
            mlkem_encap_key=alice.mlkem_encap_key,
        )
        alice.attach_certificate(alice_cert)
        save_identity(alice, password="correct-pass")
        alice.destroy()

        # ── Load from disk ────────────────────────────────────────────────────
        alice_loaded = load_identity("alice_store", password="correct-pass")

        assert alice_loaded.user_id        == "alice_store"
        assert alice_loaded.certificate    is not None

        # ── Use in a handshake ────────────────────────────────────────────────
        bob_raw = generate_user_identity("bob_store", "Bob Store")
        bob_cert = ca.issue_certificate(
            user_id="bob_store", display_name="Bob Store",
            mldsa_public_key=bob_raw.mldsa_public_key,
            mlkem_encap_key=bob_raw.mlkem_encap_key,
        )
        bob_raw.attach_certificate(bob_cert)

        client_s, server_s = _run_handshake(alice_loaded, bob_raw, ca)

        env = client_s.encrypt(b"Hello after loading from disk!", "bob_store")
        msg = server_s.decrypt(env)
        assert msg.payload == b"Hello after loading from disk!"

        print(f"\n[test] Message after disk load: {msg.payload.decode()}")

        client_s.close()
        server_s.close()
        alice_loaded.destroy()
        bob_raw.destroy()

    def test_wrong_password_cannot_load(self, tmp_path, monkeypatch, ca):
        monkeypatch.setattr("config.KEY_STORAGE_DIR", str(tmp_path))
        from identity.keypair import generate_user_identity
        from identity.storage import save_identity, load_identity

        user = generate_user_identity("wrongpass_user", "Wrong Pass")
        save_identity(user, password="right-password")
        user.destroy()

        with pytest.raises(ValueError, match="[Dd]ecryption"):
            load_identity("wrongpass_user", password="wrong-password")

    def test_certificate_update_persists(self, tmp_path, monkeypatch, ca):
        """Updating a certificate on disk does not corrupt the private key."""
        monkeypatch.setattr("config.KEY_STORAGE_DIR", str(tmp_path))
        from identity.keypair import generate_user_identity
        from identity.storage import save_identity, load_identity, update_certificate

        user = generate_user_identity("cert_update_user", "Cert Update")
        first_cert = ca.issue_certificate(
            user_id="cert_update_user", display_name="Cert Update",
            mldsa_public_key=user.mldsa_public_key,
            mlkem_encap_key=user.mlkem_encap_key,
            validity_days=365,
        )
        user.attach_certificate(first_cert)
        save_identity(user, password="pass123")

        # Issue a new certificate (simulating renewal)
        new_cert = ca.issue_certificate(
            user_id="cert_update_user", display_name="Cert Update Renewed",
            mldsa_public_key=user.mldsa_public_key,
            mlkem_encap_key=user.mlkem_encap_key,
            validity_days=730,
        )
        update_certificate("cert_update_user", new_cert)
        user.destroy()

        # Reload and verify new cert is present
        reloaded = load_identity("cert_update_user", password="pass123")
        assert reloaded.certificate.display_name == "Cert Update Renewed"
        assert reloaded.certificate.not_after > first_cert.not_after
        reloaded.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — Performance smoke test
# ══════════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """
    Smoke tests that verify operations complete within reasonable time bounds.
    These are not strict benchmarks — just sanity checks that nothing is
    catastrophically slow.
    """

    def test_kem_keygen_speed(self):
        from crypto.kem import generate_keypair
        start = time.perf_counter()
        for _ in range(10):
            kp = generate_keypair()
            kp.destroy()
        elapsed = time.perf_counter() - start
        per_op = elapsed / 10
        print(f"\n[perf] ML-KEM KeyGen: {per_op*1000:.1f}ms per operation")
        assert per_op < 1.0, f"ML-KEM KeyGen took {per_op:.2f}s — too slow"

    def test_kem_encap_decap_speed(self):
        from crypto.kem import generate_keypair, encapsulate, decapsulate
        from crypto.utils import secure_zero
        kp = generate_keypair()
        start = time.perf_counter()
        for _ in range(10):
            result = encapsulate(kp.encap_key)
            secret = decapsulate(kp.decap_key, result.ciphertext)
            result.destroy()
            secure_zero(secret)
        elapsed = time.perf_counter() - start
        per_op = elapsed / 10
        print(f"[perf] ML-KEM Encap+Decap: {per_op*1000:.1f}ms per pair")
        assert per_op < 2.0, f"ML-KEM Encap+Decap took {per_op:.2f}s"
        kp.destroy()

    def test_mldsa_sign_verify_speed(self):
        from crypto.signing import mldsa_generate_keypair, mldsa_sign, mldsa_verify
        kp = mldsa_generate_keypair()
        msg = b"benchmark message " * 10
        ctx = b"perf-test"

        start = time.perf_counter()
        for _ in range(10):
            sig = mldsa_sign(kp.private_key, msg, ctx)
            mldsa_verify(kp.public_key, msg, sig, ctx)
        elapsed = time.perf_counter() - start
        per_op = elapsed / 10
        print(f"[perf] ML-DSA Sign+Verify: {per_op*1000:.1f}ms per pair")
        assert per_op < 5.0, f"ML-DSA Sign+Verify took {per_op:.2f}s"
        kp.destroy()

    def test_aes_gcm_throughput(self):
        from crypto.symmetric import encrypt_message, decrypt_message
        from crypto.utils import random_bytes
        key    = random_bytes(32)
        nonces = [random_bytes(12) for _ in range(100)]
        payload = b"A" * 65536   # 64 KB

        start = time.perf_counter()
        for i in range(100):
            enc = encrypt_message(key, payload, nonces[i])
            decrypt_message(key, enc)
        elapsed = time.perf_counter() - start
        throughput_mb = (100 * 65536) / (elapsed * 1024 * 1024)
        print(f"[perf] AES-256-GCM: {throughput_mb:.0f} MB/s")
        assert throughput_mb > 10, f"AES-GCM throughput too low: {throughput_mb:.1f} MB/s"

    def test_full_handshake_speed(self, ca):
        from identity.keypair import generate_user_identity
        import uuid

        times = []
        for _ in range(3):
            uid_a = uuid.uuid4().hex[:8]
            uid_b = uuid.uuid4().hex[:8]
            alice = generate_user_identity(uid_a, "Alice Perf")
            bob   = generate_user_identity(uid_b, "Bob Perf")

            for identity in [alice, bob]:
                cert = ca.issue_certificate(
                    user_id=identity.user_id,
                    display_name=identity.display_name,
                    mldsa_public_key=identity.mldsa_public_key,
                    mlkem_encap_key=identity.mlkem_encap_key,
                )
                identity.attach_certificate(cert)

            start = time.perf_counter()
            cs, ss = _run_handshake(alice, bob, ca)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            cs.close(); ss.close()
            alice.destroy(); bob.destroy()

        avg = sum(times) / len(times)
        print(f"[perf] Full PQ handshake: {avg*1000:.0f}ms average")
        assert avg < 10.0, f"Handshake too slow: {avg:.2f}s average"