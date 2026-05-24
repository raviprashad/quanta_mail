"""
crypto/utils.py
───────────────
Low-level security utilities used by every other module.

Key concerns addressed here:
  • Secure deletion of secret material from memory (FIPS 203 §3.3 requirement).
  • Constant-time comparison to prevent timing side-channels.
  • Cryptographically secure random byte generation.
  • Serialisation helpers.
"""

import os
import ctypes
import secrets
import hashlib
import struct
import hmac
from typing import Union


# ─── Secure memory wiping ─────────────────────────────────────────────────────

def secure_zero(data: bytearray) -> None:
    """
    Overwrite every byte in a bytearray with zeros.

    FIPS 203 §3.3 states: "All other data shall be destroyed prior to
    the algorithm terminating."  Python's garbage collector gives no
    guarantees about when memory is reclaimed, so we zero it ourselves
    before releasing the reference.

    Usage:
        key = bytearray(secret_bytes)
        # ... use key ...
        secure_zero(key)   # always call this when done

    Note: pass bytearray, not bytes. bytes objects are immutable in
    Python so we cannot overwrite them. Always store secret material
    in bytearray when you need to guarantee deletion.
    """
    if not isinstance(data, bytearray):
        raise TypeError("secure_zero requires bytearray, got " + type(data).__name__)

    length = len(data)
    if length == 0:
        return

    # Overwrite via ctypes to prevent the compiler/interpreter from
    # optimising the write away (as it might with a pure-Python loop).
    # We use from_buffer to safely obtain a pointer to the internal buffer.
    try:
        buf = (ctypes.c_char * length).from_buffer(data)
        ctypes.memset(buf, 0, length)
    except Exception:
        # Fallback if from_buffer fails
        for i in range(length):
            data[i] = 0

    # Belt-and-suspenders: also zero via Python to handle any offset edge cases.
    for i in range(length):
        data[i] = 0


def secure_zero_bytes(data: bytes) -> None:
    """
    Best-effort zeroing for bytes objects (immutable in Python).

    This cannot guarantee the memory is overwritten because Python may
    have made internal copies.  Use bytearray for all secret material
    when possible.  This function exists only as a last resort when you
    receive a secret inside a bytes object you cannot control.
    """
    # We cannot truly zero immutable bytes in Python, but we can
    # try to overwrite the buffer via ctypes.
    # We skip singletons/interned bytes (length <= 1) to avoid corrupting CPython.
    if len(data) <= 1:
        return
    try:
        ctypes.memset(id(data) + bytes.__basicsize__, 0, len(data))
    except Exception:
        pass  # If it fails, the GC will clean up eventually.


# ─── Constant-time comparison ─────────────────────────────────────────────────

def constant_time_compare(a: bytes, b: bytes) -> bool:
    """
    Compare two byte strings in constant time.

    Regular == short-circuits on the first differing byte, leaking
    information about how many bytes match via timing.  Use this
    whenever comparing MACs, signatures, or key hashes.

    Uses hmac.compare_digest which is guaranteed constant-time by
    the Python standard library.
    """
    return hmac.compare_digest(a, b)


# ─── Secure random generation ─────────────────────────────────────────────────

def random_bytes(n: int) -> bytes:
    """
    Generate n cryptographically secure random bytes.

    Uses os.urandom which calls the OS CSPRNG (getrandom() on Linux,
    CryptGenRandom on Windows, arc4random on macOS/BSD).

    FIPS 203 §3.3 requires an approved RBG.  os.urandom satisfies
    this requirement on FIPS-certified operating systems.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return os.urandom(n)


def random_token(n: int = 32) -> str:
    """Return a URL-safe base64 random token of n bytes entropy."""
    return secrets.token_urlsafe(n)


# ─── Hashing helpers ──────────────────────────────────────────────────────────

def sha3_256(data: bytes) -> bytes:
    """SHA3-256 hash. Used for public key fingerprints and session IDs."""
    return hashlib.sha3_256(data).digest()


def sha3_512(data: bytes) -> bytes:
    """SHA3-512 hash. Used for transcript hashing in the handshake."""
    return hashlib.sha3_512(data).digest()


# ─── Transcript hashing ───────────────────────────────────────────────────────

class TranscriptHasher:
    """
    Incrementally hash all handshake messages into a single digest.

    Both sides of a handshake build the same transcript and sign/verify
    it.  If any byte of any message was tampered with, the transcripts
    will differ and authentication fails.

    Usage:
        t = TranscriptHasher()
        t.add(b"ClientHello bytes")
        t.add(b"ServerHello bytes")
        digest = t.digest()   # SHA3-512 of everything added so far
    """

    def __init__(self):
        self._h = hashlib.sha3_512()

    def add(self, data: bytes) -> None:
        """Feed bytes into the running hash."""
        # Length-prefix each chunk so that ("ab", "c") ≠ ("a", "bc").
        self._h.update(struct.pack(">I", len(data)))
        self._h.update(data)

    def digest(self) -> bytes:
        """Return the current 64-byte digest without finalising."""
        return self._h.copy().digest()


# ─── Serialisation helpers ────────────────────────────────────────────────────

def encode_length_prefixed(data: bytes) -> bytes:
    """Prefix data with a 4-byte big-endian length field."""
    return struct.pack(">I", len(data)) + data


def decode_length_prefixed(buf: bytes, offset: int = 0) -> tuple[bytes, int]:
    """
    Read a length-prefixed chunk from buf starting at offset.
    Returns (chunk, new_offset).
    """
    if offset + 4 > len(buf):
        raise ValueError("Buffer too short to read length prefix")
    length = struct.unpack_from(">I", buf, offset)[0]
    offset += 4
    if offset + length > len(buf):
        raise ValueError(f"Buffer too short: need {length} bytes, have {len(buf) - offset}")
    chunk = buf[offset: offset + length]
    return chunk, offset + length


def public_key_fingerprint(public_key_bytes: bytes) -> str:
    """
    Return a human-readable hex fingerprint of a public key.
    Used to identify keys in logs and UI without exposing the full key.
    """
    digest = sha3_256(public_key_bytes)
    # Format as colon-separated hex pairs: "AB:CD:EF:..."
    return ":".join(f"{b:02X}" for b in digest[:16])