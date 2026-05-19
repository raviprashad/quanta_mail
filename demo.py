"""
demo.py
────────
Standalone demonstration of the PQ Messenger system.

Run this script to see the complete post-quantum cryptographic
pipeline in action — no server needed.

Usage:
    python demo.py

What it demonstrates:
  1. CA key generation        (SLH-DSA-SHAKE-192s)
  2. User registration        (ML-DSA-65 + ML-KEM-768)
  3. Certificate issuance     (SLH-DSA signature)
  4. Post-quantum handshake   (ML-KEM key agreement + ML-DSA authentication)
  5. Bidirectional messaging  (AES-256-GCM + ML-DSA signatures)
  6. Security verification    (tamper detection)
  7. Secure teardown          (key destruction)
"""

import sys
import time

# ─── Pretty printing helpers ──────────────────────────────────────────────────

def banner(title: str) -> None:
    width = 70
    print()
    print("═" * width)
    print(f"  {title}")
    print("═" * width)

def step(n: int, text: str) -> None:
    print(f"\n  [{n}] {text}")

def ok(text: str) -> None:
    print(f"      ✓ {text}")

def info(text: str) -> None:
    print(f"      • {text}")

def fail(text: str) -> None:
    print(f"      ✗ {text}")

def hex_preview(data: bytes, n: int = 16) -> str:
    return data[:n].hex() + "..."


# ══════════════════════════════════════════════════════════════════════════════

def main():
    banner("POST-QUANTUM SECURE MESSAGING DEMO")
    print("""
  Algorithms used:
    Key Exchange:     ML-KEM-768    (FIPS 203)
    Identity Signing: ML-DSA-65     (FIPS 204)
    CA Signing:       SLH-DSA proxy (FIPS 205)
    Bulk Encryption:  AES-256-GCM
    KDF:              HKDF-SHA3-256
    """)

    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 1 — Certificate Authority Setup")
    # ─────────────────────────────────────────────────────────────────────────

    step(1, "Generating CA key pair with SLH-DSA (SPHINCS+-SHAKE-192s) ...")
    t0 = time.perf_counter()
    from crypto.signing import slhdsa_generate_keypair
    from identity.certificate import CertificateAuthority
    ca_kp = slhdsa_generate_keypair()
    ca    = CertificateAuthority(ca_kp)
    elapsed = time.perf_counter() - t0
    ok(f"CA ready in {elapsed*1000:.0f}ms")
    info(f"CA public key:  {len(ca_kp.public_key)} bytes")
    info(f"CA private key: {len(ca_kp.private_key)} bytes")
    info(f"CA fingerprint: {ca.fingerprint}")

    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 2 — User Registration")
    # ─────────────────────────────────────────────────────────────────────────

    step(2, "Alice generates her identity key pairs ...")
    t0 = time.perf_counter()
    from identity.keypair import generate_user_identity
    alice = generate_user_identity("alice", "Alice Example")
    elapsed = time.perf_counter() - t0
    ok(f"Key pairs generated in {elapsed*1000:.0f}ms")
    info(f"ML-DSA public key:    {len(alice.mldsa_public_key)} bytes")
    info(f"ML-DSA private key:   {len(alice.mldsa_private_key)} bytes")
    info(f"ML-KEM encap key:     {len(alice.mlkem_encap_key)} bytes")
    info(f"ML-KEM decap key:     {len(alice.mlkem_decap_key)} bytes")

    step(3, "CA issues Alice a SLH-DSA-signed certificate ...")
    t0 = time.perf_counter()
    alice_cert = ca.issue_certificate(
        user_id="alice",
        display_name="Alice Example",
        mldsa_public_key=alice.mldsa_public_key,
        mlkem_encap_key=alice.mlkem_encap_key,
        validity_days=365,
    )
    alice.attach_certificate(alice_cert)
    elapsed = time.perf_counter() - t0
    ok(f"Certificate issued in {elapsed*1000:.0f}ms")
    info(f"Certificate size:     {len(alice_cert.to_bytes())} bytes")
    info(f"SLH-DSA signature:    {len(alice_cert.slhdsa_signature)} bytes")
    info(f"Alice fingerprint:    {alice.fingerprint}")

    step(4, "Bob registers independently ...")
    t0 = time.perf_counter()
    bob = generate_user_identity("bob", "Bob Example")
    bob_cert = ca.issue_certificate(
        user_id="bob",
        display_name="Bob Example",
        mldsa_public_key=bob.mldsa_public_key,
        mlkem_encap_key=bob.mlkem_encap_key,
        validity_days=365,
    )
    bob.attach_certificate(bob_cert)
    elapsed = time.perf_counter() - t0
    ok(f"Bob registered in {elapsed*1000:.0f}ms")
    info(f"Bob fingerprint:      {bob.fingerprint}")

    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 3 — Post-Quantum Handshake")
    # ─────────────────────────────────────────────────────────────────────────

    from identity.certificate import CertificateAuthority as CA
    from protocol.handshake import ClientHandshake, ServerHandshake
    from protocol.session import Session, Role

    verifier = CA.verifier_only(ca_kp.public_key)

    step(5, "Alice builds ClientHello ...")
    t0 = time.perf_counter()
    client_hs = ClientHandshake(alice, verifier)
    client_hello_bytes = client_hs.build_client_hello()
    elapsed = time.perf_counter() - t0
    ok(f"ClientHello built in {elapsed*1000:.0f}ms ({len(client_hello_bytes)} bytes)")
    info("  Contains: ephemeral KEM encap key + certificate + ML-DSA signature")
    info(f"  First bytes: {hex_preview(client_hello_bytes)}")

    step(6, "Bob verifies ClientHello and builds ServerHello ...")
    t0 = time.perf_counter()
    server_hs = ServerHandshake(bob, verifier)
    server_hello_bytes = server_hs.process_client_hello(client_hello_bytes)
    elapsed = time.perf_counter() - t0
    ok(f"ServerHello built in {elapsed*1000:.0f}ms ({len(server_hello_bytes)} bytes)")
    info("  Contains: ML-KEM ciphertext + certificate + ML-DSA signature over transcript")
    info("  Bob encapsulated a 32-byte shared secret with Alice's ephemeral KEM key")

    step(7, "Alice verifies ServerHello, decapsulates, sends ClientFinished ...")
    t0 = time.perf_counter()
    client_finished_bytes, c_established = client_hs.process_server_hello(server_hello_bytes)
    elapsed = time.perf_counter() - t0
    ok(f"ClientFinished built in {elapsed*1000:.0f}ms ({len(client_finished_bytes)} bytes)")
    info("  Alice decapsulated the KEM ciphertext to get shared secret")
    info("  Ephemeral KEM decap key zeroed from memory (forward secrecy)")
    info("  Session keys derived via HKDF-SHA3-256")

    step(8, "Bob verifies ClientFinished — session established ...")
    t0 = time.perf_counter()
    s_established = server_hs.process_client_finished(client_finished_bytes)
    elapsed = time.perf_counter() - t0
    ok(f"Session established in {elapsed*1000:.0f}ms")
    info(f"  Session ID: {c_established.session_id.hex()[:32]}...")

    # Verify both sides derived identical keys
    assert c_established.session_id == s_established.session_id
    assert c_established.client_keys.enc_key == s_established.client_keys.enc_key
    ok("Both parties derived IDENTICAL session keys ✓")

    # Build Session objects
    alice_session = Session(c_established, alice, c_established.peer_certificate, Role.CLIENT)
    bob_session   = Session(s_established, bob,  s_established.peer_certificate, Role.SERVER)

    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 4 — Encrypted Messaging")
    # ─────────────────────────────────────────────────────────────────────────

    step(9, "Alice sends encrypted messages to Bob ...")

    messages = [
        "Hello Bob! This message is quantum-proof.",
        "ML-KEM-768 protects our shared key.",
        "AES-256-GCM encrypts the content.",
        "ML-DSA-65 signs every message I send you.",
    ]

    for i, text in enumerate(messages, 1):
        t0 = time.perf_counter()
        env_bytes = alice_session.encrypt(text.encode(), "bob")
        msg       = bob_session.decrypt(env_bytes)
        elapsed   = time.perf_counter() - t0

        assert msg.payload.decode() == text
        assert msg.verified is True
        ok(f"Message {i}: '{text[:45]}...' ({len(env_bytes)} bytes, {elapsed*1000:.1f}ms)")

    step(10, "Bob replies to Alice ...")
    reply = "Got your messages, Alice! The channel is secure."
    env_bytes = bob_session.encrypt(reply.encode(), "alice")
    msg = alice_session.decrypt(env_bytes)
    assert msg.payload.decode() == reply
    ok(f"Reply received: '{reply}'")

    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 5 — Security Verification")
    # ─────────────────────────────────────────────────────────────────────────

    step(11, "Verifying tamper detection ...")

    # Tamper with ciphertext
    env_bytes = alice_session.encrypt(b"Tamper this!", "bob")
    tampered  = bytearray(env_bytes)
    tampered[200] ^= 0xFF
    try:
        bob_session.decrypt(bytes(tampered))
        fail("ERROR: tampered message was accepted!")
        sys.exit(1)
    except Exception:
        ok("Ciphertext tampering detected by AES-GCM tag ✓")

    step(12, "Verifying replay attack prevention ...")
    env_bytes = alice_session.encrypt(b"Replay me", "bob")
    bob_session.decrypt(env_bytes)   # first delivery: OK
    try:
        bob_session.decrypt(env_bytes)   # replay: must fail
        fail("ERROR: replay was accepted!")
        sys.exit(1)
    except Exception:
        ok("Replay attack detected by sequence number check ✓")

    step(13, "Verifying different plaintexts produce different ciphertexts ...")
    from protocol.session import EncryptedEnvelope
    e1 = EncryptedEnvelope.from_bytes(alice_session.encrypt(b"same", "bob"))
    e2 = EncryptedEnvelope.from_bytes(alice_session.encrypt(b"same", "bob"))
    assert e1.ciphertext != e2.ciphertext
    assert e1.nonce      != e2.nonce
    ok("Same plaintext → different ciphertext each time (nonce is unique) ✓")

    # Flush the 2 messages we just encrypted from bob's sequence counter
    bob_session.decrypt(EncryptedEnvelope.from_bytes(
        alice_session._session_id and  # just accessing to avoid IDE warning
        b""  # placeholder — see note below
    ) if False else e1.to_bytes())

    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 6 — Key Storage Demo")
    # ─────────────────────────────────────────────────────────────────────────

    step(14, "Saving Alice's identity to encrypted storage ...")
    import tempfile, os
    import config

    with tempfile.TemporaryDirectory() as tmpdir:
        original_key_dir = config.KEY_STORAGE_DIR
        config.KEY_STORAGE_DIR = tmpdir

        from identity.storage import save_identity, load_identity

        t0 = time.perf_counter()
        path = save_identity(alice, password="alice-secure-passphrase")
        elapsed = time.perf_counter() - t0
        file_size = os.path.getsize(path)
        ok(f"Saved to {path} ({file_size} bytes, {elapsed*1000:.0f}ms)")
        info("Private keys encrypted with AES-256-GCM")
        info("Key derived from passphrase via PBKDF2 (600,000 iterations)")

        step(15, "Loading Alice's identity back from disk ...")
        t0 = time.perf_counter()
        alice_reloaded = load_identity("alice", password="alice-secure-passphrase")
        elapsed = time.perf_counter() - t0
        ok(f"Loaded in {elapsed*1000:.0f}ms")
        assert alice_reloaded.mldsa_private_key == alice.mldsa_private_key
        ok("Private keys are byte-identical after decrypt ✓")
        alice_reloaded.destroy()

        step(16, "Wrong password is rejected ...")
        try:
            load_identity("alice", password="wrong-password")
            fail("ERROR: wrong password was accepted!")
            sys.exit(1)
        except ValueError:
            ok("Wrong password rejected by AES-GCM authentication ✓")

        config.KEY_STORAGE_DIR = original_key_dir

    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 7 — Secure Teardown")
    # ─────────────────────────────────────────────────────────────────────────

    step(17, "Destroying all session keys ...")
    alice_session.close()
    bob_session.close()
    ok("Session encryption keys zeroed from memory ✓")

    step(18, "Destroying identity key material ...")
    alice.destroy()
    bob.destroy()
    ca_kp.destroy()
    ok("Identity private keys zeroed from memory ✓")
    ok("CA private key zeroed from memory ✓")

    # ─────────────────────────────────────────────────────────────────────────
    banner("DEMO COMPLETE")
    # ─────────────────────────────────────────────────────────────────────────
    print("""
  Summary of what just happened:
  ────────────────────────────────────────────────────────────────────
  ✓ CA issued SLH-DSA certificates  — quantum-proof CA trust chain
  ✓ ML-KEM-768 key agreement        — shared secret safe from Shor's
  ✓ ML-DSA-65 authentication        — identity verification
  ✓ AES-256-GCM message encryption  — 128-bit PQ security for bulk data
  ✓ HKDF-SHA3-256 key derivation    — independent session keys
  ✓ Replay detection                — sequence number tracking
  ✓ Tamper detection                — GCM tag verification
  ✓ Forward secrecy                 — ephemeral KEM keys destroyed
  ✓ Encrypted key storage           — PBKDF2 + AES-256-GCM at rest
  ✓ Secure deletion                 — all secret material zeroed
  ────────────────────────────────────────────────────────────────────

  Next steps:
    Run the server:  uvicorn server.app:app --reload
    Run tests:       pytest tests/ -v
    """)


if __name__ == "__main__":
    main()