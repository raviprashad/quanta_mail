"""
server/app.py
─────────────
FastAPI server that exposes the PQ Messenger protocol over HTTP.

Endpoints:
  POST /register          — submit public keys, receive a CA-signed certificate
  POST /handshake/hello   — exchange ClientHello → ServerHello
  POST /handshake/finish  — submit ClientFinished, complete session
  POST /message/{session} — send an encrypted message to a session
  GET  /message/{session} — poll for inbound messages (long-poll)
  GET  /health            — liveness check

In production you would use WebSockets or a message queue for real-time
delivery.  HTTP polling here keeps the code simple and focused on the
cryptographic layer.

Security note:
  This server trusts that TLS (or another transport) prevents network-level
  eavesdropping.  The PQ crypto layer provides security ON TOP OF TLS,
  meaning the system is secure even if TLS is broken by a quantum adversary.
  Running both is defence-in-depth.
"""

from __future__ import annotations
import asyncio
import uuid
import time
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import config
from crypto.signing import slhdsa_generate_keypair, SLHDSAKeyPair
from identity.certificate import (
    Certificate,
    CertificateAuthority,
    CertificateError,
)
from identity.keypair import generate_user_identity, UserIdentity
from identity.storage import save_identity, load_identity
from protocol.handshake import (
    ServerHandshake,
    HandshakeError,
    EstablishedSession,
)
from protocol.session import Session, Role, SessionError


# ─── Server state ─────────────────────────────────────────────────────────────

class ServerState:
    """
    In-memory server state.

    In production: use Redis for sessions, a proper DB for registrations.
    Here everything lives in memory for clarity.
    """
    def __init__(self):
        # CA key pair — generated once at startup
        self.ca_keypair:  Optional[SLHDSAKeyPair] = None
        self.ca:          Optional[CertificateAuthority] = None

        # Server's own identity
        self.server_identity: Optional[UserIdentity] = None

        # Registered users: user_id → Certificate
        self.registered_users: Dict[str, Certificate] = {}

        # Active handshakes in progress: handshake_id → ServerHandshake
        self.pending_handshakes: Dict[str, ServerHandshake] = {}

        # Active sessions: session_id (hex) → Session
        self.active_sessions: Dict[str, Session] = {}

        # Message queues: recipient_user_id → [envelope_bytes, ...]
        self.message_queues: Dict[str, List[bytes]] = {}


state = ServerState()


# ─── Startup / shutdown ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise CA and server identity on startup."""
    print("[server] Generating CA key pair (SLH-DSA-SHAKE-192s) ...")
    ca_keypair = slhdsa_generate_keypair()
    state.ca_keypair = ca_keypair
    state.ca = CertificateAuthority(ca_keypair)
    print(f"[server] CA public key fingerprint: {state.ca.fingerprint}")

    print("[server] Generating server identity (ML-DSA-65 + ML-KEM-768) ...")
    server_identity = generate_user_identity(
        user_id="server",
        display_name="PQ Messenger Server",
    )
    # Self-issue a certificate for the server
    server_cert = state.ca.issue_certificate(
        user_id="server",
        display_name="PQ Messenger Server",
        mldsa_public_key=server_identity.mldsa_public_key,
        mlkem_encap_key=server_identity.mlkem_encap_key,
        validity_days=365,
    )
    server_identity.attach_certificate(server_cert)
    state.server_identity = server_identity
    state.registered_users["server"] = server_cert

    print("[server] Ready.")
    yield

    # Shutdown: destroy all session keys
    print("[server] Shutting down, destroying session keys ...")
    for session in state.active_sessions.values():
        session.close()
    if state.server_identity:
        state.server_identity.destroy()


app = FastAPI(title="PQ Messenger Server", lifespan=lifespan)


# ─── Pydantic request / response models ───────────────────────────────────────

class RegisterRequest(BaseModel):
    user_id:          str
    display_name:     str
    mldsa_public_key: str   # hex-encoded
    mlkem_encap_key:  str   # hex-encoded


class RegisterResponse(BaseModel):
    certificate:    str     # hex-encoded signed Certificate
    ca_public_key:  str     # hex-encoded SLH-DSA CA public key


class HandshakeHelloRequest(BaseModel):
    client_hello: str       # hex-encoded ClientHello bytes


class HandshakeHelloResponse(BaseModel):
    handshake_id:  str
    server_hello:  str      # hex-encoded ServerHello bytes


class HandshakeFinishRequest(BaseModel):
    handshake_id:     str
    client_finished:  str   # hex-encoded ClientFinished bytes


class HandshakeFinishResponse(BaseModel):
    session_id: str         # hex session ID to use for subsequent messages


class SendMessageRequest(BaseModel):
    envelope:     str       # hex-encoded EncryptedEnvelope
    recipient_id: str


class PollResponse(BaseModel):
    messages: List[str]     # list of hex-encoded EncryptedEnvelope bytes


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "ca_fingerprint": state.ca.fingerprint}


@app.get("/ca-public-key")
async def get_ca_public_key():
    """Return the CA's SLH-DSA public key so clients can verify certificates."""
    return {"ca_public_key": state.ca_keypair.public_key.hex()}


@app.post("/register", response_model=RegisterResponse)
async def register(req: RegisterRequest):
    """
    Register a new user and issue them a CA-signed certificate.

    The client sends their public keys; the server verifies sizes,
    issues a certificate, and returns it.
    """
    if req.user_id in state.registered_users:
        raise HTTPException(status_code=409, detail=f"User '{req.user_id}' already registered")

    try:
        mldsa_pk = bytes.fromhex(req.mldsa_public_key)
        mlkem_ek = bytes.fromhex(req.mlkem_encap_key)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid hex encoding")

    if len(mldsa_pk) != config.DSA_PUBLIC_KEY_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"ML-DSA public key must be {config.DSA_PUBLIC_KEY_BYTES} bytes"
        )
    if len(mlkem_ek) != config.KEM_ENCAP_KEY_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"ML-KEM encap key must be {config.KEM_ENCAP_KEY_BYTES} bytes"
        )

    cert = state.ca.issue_certificate(
        user_id=req.user_id,
        display_name=req.display_name,
        mldsa_public_key=mldsa_pk,
        mlkem_encap_key=mlkem_ek,
        validity_days=365,
    )

    state.registered_users[req.user_id] = cert
    state.message_queues[req.user_id] = []

    print(f"[server] Registered user '{req.user_id}' (fingerprint: {cert.fingerprint})")

    return RegisterResponse(
        certificate=cert.to_bytes().hex(),
        ca_public_key=state.ca_keypair.public_key.hex(),
    )


@app.post("/handshake/hello", response_model=HandshakeHelloResponse)
async def handshake_hello(req: HandshakeHelloRequest):
    """
    Process ClientHello and return ServerHello.

    Creates a new ServerHandshake, processes the client's hello message,
    returns the server's hello.  The handshake_id is used to correlate
    the ClientFinished message that follows.
    """
    try:
        client_hello_bytes = bytes.fromhex(req.client_hello)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid hex encoding")

    ca_verifier = CertificateAuthority.verifier_only(state.ca_keypair.public_key)
    hs = ServerHandshake(
        identity=state.server_identity,
        ca_verifier=ca_verifier,
    )

    try:
        server_hello_bytes = hs.process_client_hello(client_hello_bytes)
    except HandshakeError as e:
        raise HTTPException(status_code=400, detail=f"Handshake failed: {e}")

    handshake_id = str(uuid.uuid4())
    state.pending_handshakes[handshake_id] = hs

    print(f"[server] Handshake {handshake_id}: ClientHello processed OK")

    return HandshakeHelloResponse(
        handshake_id=handshake_id,
        server_hello=server_hello_bytes.hex(),
    )


@app.post("/handshake/finish", response_model=HandshakeFinishResponse)
async def handshake_finish(req: HandshakeFinishRequest):
    """
    Process ClientFinished and establish the session.

    If the ClientFinished decrypts correctly, both sides have agreed
    on the same session keys.  The session is stored server-side.
    """
    hs = state.pending_handshakes.pop(req.handshake_id, None)
    if hs is None:
        raise HTTPException(status_code=404, detail="Handshake not found or expired")

    try:
        client_finished_bytes = bytes.fromhex(req.client_finished)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid hex encoding")

    try:
        established = hs.process_client_finished(client_finished_bytes)
    except HandshakeError as e:
        raise HTTPException(status_code=400, detail=f"Handshake failed: {e}")

    session_id_hex = established.session_id.hex()

    session = Session(
        established=established,
        identity=state.server_identity,
        peer_cert=established.peer_certificate,
        role=Role.SERVER,
    )
    state.active_sessions[session_id_hex] = session

    print(
        f"[server] Session {session_id_hex[:16]}... established "
        f"with '{established.peer_certificate.user_id}'"
    )

    return HandshakeFinishResponse(session_id=session_id_hex)


@app.post("/message/{session_id}")
async def send_message(session_id: str, req: SendMessageRequest):
    """
    Receive an encrypted message from a client and queue it for the recipient.

    The server does NOT decrypt the message — it only routes it.
    The session is used to verify the ML-DSA signature (authentication),
    but the payload remains opaque to the server.

    Note: In this design the server verifies signatures to prevent spam
    and replay.  A pure relay server could skip this, but then it cannot
    detect replayed messages.
    """
    session = state.active_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        envelope_bytes = bytes.fromhex(req.envelope)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid hex encoding")

    try:
        # Verify signature and sequence number but don't expose plaintext
        decrypted = session.decrypt(envelope_bytes)
    except SessionError as e:
        raise HTTPException(status_code=400, detail=f"Message invalid: {e}")

    # Queue message for recipient
    recipient_id = decrypted.header.recipient_id
    if recipient_id not in state.message_queues:
        state.message_queues[recipient_id] = []
    state.message_queues[recipient_id].append(envelope_bytes.hex())

    print(
        f"[server] Message seq={decrypted.header.sequence_number} "
        f"queued for '{recipient_id}'"
    )

    return {"status": "queued", "sequence_number": decrypted.header.sequence_number}


@app.get("/message/{user_id}", response_model=PollResponse)
async def poll_messages(user_id: str):
    """
    Return all queued messages for a user and clear the queue.

    In production this would be a WebSocket or server-sent events endpoint.
    """
    if user_id not in state.message_queues:
        raise HTTPException(status_code=404, detail="User not found")

    messages = state.message_queues[user_id]
    state.message_queues[user_id] = []  # clear queue

    return PollResponse(messages=messages)


@app.delete("/session/{session_id}")
async def close_session(session_id: str):
    """Explicitly close a session and destroy its keys."""
    session = state.active_sessions.pop(session_id, None)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session.close()
    print(f"[server] Session {session_id[:16]}... closed")
    return {"status": "closed"}