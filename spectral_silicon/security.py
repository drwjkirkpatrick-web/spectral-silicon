"""Bitstream encryption and integrity hashing for the Spectral Silicon chip.

This module implements the Python-side counterparts of two RTL modules:

* **weight_crypto.v** (Improvement 11) — an LFSR-based stream cipher that
  decrypts weight bitstreams on-the-fly so the external bus never sees
  plaintext weights after manufacturing.
* **integrity_hash.v** (Improvement 11) — an FNV-1a 32-bit rolling hash that
  verifies weight-blob integrity at load time.

The :class:`WeightBitstream` class combines both primitives into a single
encrypt-and-hash workflow that produces a tamper-evident encrypted blob.

LFSR stream cipher
-------------------
The cipher uses a 32-bit Galois-type LFSR with the CRC-32 polynomial
``x^32 + x^22 + x^2 + 1`` (taps at bits 0, 2, 22).  The 32-bit key seeds
the LFSR; each cycle the full 32-bit state is advanced one step and the
resulting 32-bit word is XORed with one 32-bit word of plaintext to
produce ciphertext (and vice-versa, since the cipher is symmetric).

::

    LFSR polynomial:  x^32 + x^22 + x^2 + 1
    Taps (Galois):    bits 0, 2, 22
    Period:            2^32 - 1  (maximal-length)

FNV-1a 32-bit hash
------------------
The FNV-1a hash processes input bytes one at a time::

    hash = FNV_OFFSET_BASIS_32
    for byte in data:
        hash = hash XOR byte
        hash = hash * FNV_PRIME_32
    return hash & 0xFFFFFFFF

This matches the Verilog ``integrity_hash.v`` rolling hash.

Examples
--------
>>> from spectral_silicon.security import LFSRCipher, IntegrityHash, WeightBitstream
>>> cipher = LFSRCipher(key=0xDEADBEEF)
>>> ct = cipher.encrypt(b"hello world")
>>> cipher2 = LFSRCipher(key=0xDEADBEEF)
>>> cipher2.decrypt(ct)
b'hello world'
>>> h = IntegrityHash()
>>> digest = h.compute(b"some data")
>>> h.verify(b"some data", digest)
True

>>> blob = WeightBitstream.pack(b"my weights", key=0x12345678)
>>> WeightBitstream.unpack(blob, key=0x12345678)
b'my weights'
"""

from __future__ import annotations

import struct
from typing import Union

__all__ = [
    "LFSRCipher",
    "IntegrityHash",
    "WeightBitstream",
    "TamperError",
]


# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────

# CRC-32 polynomial: x^32 + x^22 + x^2 + 1
# Galois taps at bit positions 0, 2, 22 → feedback mask 0x00400004
LFSR_POLY_TAPS = 0x00400004  # bit 22 | bit 2
LFSR_MASK = 0xFFFFFFFF  # 32-bit mask

# FNV-1a 32-bit constants (public, well-known)
FNV_OFFSET_BASIS_32 = 0x811C9DC5
FNV_PRIME_32 = 0x01000193
FNV_MASK_32 = 0xFFFFFFFF

# WeightBitstream envelope format (all little-endian):
#   4s  magic       b"WSEC"
#   I   version     1
#   I   hash         FNV-1a 32-bit hash of the *plaintext* weights
#   I   payload_len  length of the encrypted payload in bytes
#   8s  reserved    zero padding
WB_MAGIC = b"WSEC"
WB_VERSION = 1
WB_HEADER_FORMAT = "<4sIII8s"
WB_HEADER_SIZE = struct.calcsize(WB_HEADER_FORMAT)  # 4 + 3*4 + 8 = 24


class TamperError(Exception):
    """Raised when a weight bitstream fails integrity verification."""


# ──────────────────────────────────────────────────────────────────────────
# LFSRCipher — 32-bit Galois LFSR stream cipher
# ──────────────────────────────────────────────────────────────────────────


class LFSRCipher:
    """32-bit LFSR-based stream cipher matching Verilog ``weight_crypto.v``.

    The cipher is a symmetric stream cipher: encryption and decryption are
    identical operations (XOR with the LFSR keystream).  A fresh cipher
    instance must be created for each encryption/decryption session because
    the internal LFSR state advances with every byte processed.

    The LFSR is Galois-type with polynomial ``x^32 + x^22 + x^2 + 1``.
    On each step the register shifts right by one bit; if the shifted-out
    bit (former LSB) was 1, the feedback mask ``0x00400004`` is XORed into
    the register.

    Parameters
    ----------
    key : int
        32-bit unsigned integer used to seed the LFSR.  A key of ``0``
        puts the cipher into *bypass mode*: the LFSR state stays at 0
        and the keystream is all zeros, so ``encrypt`` / ``decrypt``
        return the input unchanged (matching the Verilog test mode).

    Attributes
    ----------
    key : int
        The original 32-bit key (masked to 32 bits).
    state : int
        Current 32-bit LFSR state.

    Examples
    --------
    >>> c = LFSRCipher(key=0xA5A5A5A5)
    >>> ct = c.encrypt(b"spectral weights")
    >>> c2 = LFSRCipher(key=0xA5A5A5A5)
    >>> c2.decrypt(ct) == b"spectral weights"
    True
    >>> bypass = LFSRCipher(key=0)
    >>> bypass.encrypt(b"test") == b"test"
    True
    """

    def __init__(self, key: int) -> None:
        if not isinstance(key, int):
            raise TypeError(f"key must be int, got {type(key).__name__}")
        if key < 0 or key > 0xFFFFFFFF:
            raise ValueError(
                f"key must be a 32-bit unsigned integer (0..{0xFFFFFFFF}), "
                f"got {key}"
            )
        self.key = key & LFSR_MASK
        self.state = key & LFSR_MASK

    # ── internal LFSR stepping ──────────────────────────────────────────

    def _step(self) -> int:
        """Advance the LFSR one step and return the new 32-bit state.

        Galois LFSR shift-right: the LSB is the feedback bit.  If it is 1,
        XOR the tap mask into the shifted register.
        """
        lsb = self.state & 1
        self.state >>= 1
        if lsb:
            self.state ^= LFSR_POLY_TAPS
        self.state &= LFSR_MASK
        return self.state

    def _keystream_word(self) -> int:
        """Generate one 32-bit keystream word.

        The Verilog module uses ``lfsr_next`` (the *next* state) as the
        keystream.  We replicate that: step the LFSR and use the resulting
        state as the XOR keystream.
        """
        return self._step()

    # ── public API ──────────────────────────────────────────────────────

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt *data* by XORing it with the LFSR keystream.

        Because the cipher is a symmetric stream cipher, ``decrypt`` is
        identical to ``encrypt`` — both XOR the data with the same
        keystream sequence.

        The data is processed in 32-bit (4-byte) words.  Any trailing
        bytes that do not fill a complete word are XORed with the
        least-significant bytes of a keystream word.

        Parameters
        ----------
        data : bytes
            Plaintext (for encryption) or ciphertext (for decryption).

        Returns
        -------
        bytes
            Ciphertext (for encryption) or plaintext (for decryption).
        """
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(
                f"data must be bytes or bytearray, got {type(data).__name__}"
            )
        data = bytes(data)
        if self.key == 0:
            # Bypass mode — keystream is all zeros (matches Verilog test mode)
            return data

        out = bytearray()
        n = len(data)
        n_full_words = n // 4
        # Process full 32-bit words
        for i in range(n_full_words):
            word = struct.unpack_from("<I", data, i * 4)[0]
            ks = self._keystream_word()
            cipher_word = word ^ ks
            struct.pack_into("<I", out, len(out), cipher_word) if False else None
            out.extend(struct.pack("<I", cipher_word))

        # Handle trailing partial word (1-3 bytes)
        remainder = n % 4
        if remainder:
            tail = data[n_full_words * 4 :]
            ks = self._keystream_word()
            ks_bytes = struct.pack("<I", ks)
            for j in range(remainder):
                out.append(tail[j] ^ ks_bytes[j])

        return bytes(out)

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt *data* — identical to :meth:`encrypt` (symmetric cipher).

        Parameters
        ----------
        data : bytes
            Ciphertext to decrypt.

        Returns
        -------
        bytes
            Recovered plaintext.
        """
        return self.encrypt(data)

    def keystream(self, n_words: int) -> bytes:
        """Generate *n_words* of raw keystream as bytes (for testing).

        Parameters
        ----------
        n_words : int
            Number of 32-bit keystream words to generate.

        Returns
        -------
        bytes
            ``n_words * 4`` bytes of keystream.
        """
        out = bytearray()
        for _ in range(n_words):
            out.extend(struct.pack("<I", self._keystream_word()))
        return bytes(out)

    def reset(self) -> None:
        """Reset the LFSR state back to the original key."""
        self.state = self.key & LFSR_MASK


# ──────────────────────────────────────────────────────────────────────────
# IntegrityHash — FNV-1a 32-bit rolling hash
# ──────────────────────────────────────────────────────────────────────────


class IntegrityHash:
    """FNV-1a 32-bit hash matching Verilog ``integrity_hash.v``.

    The FNV-1a (Fowler-Noll-Vo) hash processes input bytes sequentially::

        hash = 2166136261   (FNV offset basis)
        for byte in data:
            hash ^= byte
            hash *= 16777619  (FNV prime)
            hash &= 0xFFFFFFFF

    Parameters
    ----------
    init : int, optional
        Initial hash value.  Defaults to the standard FNV-1a offset basis
        ``0x811C9DC5``.  Pass a custom value only if you need to continue
        a previous hash computation.

    Attributes
    ----------
    offset_basis : int
        The FNV-1a 32-bit offset basis.
    prime : int
        The FNV-1a 32-bit prime.
    hash : int
        Current rolling hash value (32-bit).

    Examples
--------
    >>> h = IntegrityHash()
    >>> h.compute(b"") == FNV_OFFSET_BASIS_32
    True
    >>> h.compute(b"hello") != h.compute(b"world")
    True
    >>> h.verify(b"hello", h.compute(b"hello"))
    True
    """

    def __init__(self, init: int = FNV_OFFSET_BASIS_32) -> None:
        self.offset_basis = FNV_OFFSET_BASIS_32
        self.prime = FNV_PRIME_32
        self.hash = init & FNV_MASK_32

    def compute(self, data: bytes) -> int:
        """Compute the FNV-1a 32-bit hash of *data*.

        This is a *fresh* computation: the hash is re-initialized to the
        FNV offset basis before processing, so the result depends only on
        *data*, not on any prior state.

        Parameters
        ----------
        data : bytes
            Input data to hash.

        Returns
        -------
        int
            32-bit FNV-1a hash value.
        """
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(
                f"data must be bytes or bytearray, got {type(data).__name__}"
            )
        h = self.offset_basis
        for byte in data:
            h ^= byte
            h = (h * self.prime) & FNV_MASK_32
        return h

    def verify(self, data: bytes, expected: int) -> bool:
        """Verify that *data* hashes to *expected*.

        Parameters
        ----------
        data : bytes
            Input data to hash and check.
        expected : int
            The expected 32-bit FNV-1a hash value.

        Returns
        -------
        bool
            ``True`` if ``compute(data) == expected``, ``False`` otherwise.
        """
        return self.compute(data) == (expected & FNV_MASK_32)

    def update(self, byte: int) -> None:
        """Advance the rolling hash by one byte (streaming mode).

        Parameters
        ----------
        byte : int
            A single byte value (0-255).
        """
        if not 0 <= byte <= 255:
            raise ValueError(f"byte must be 0..255, got {byte}")
        self.hash ^= byte
        self.hash = (self.hash * self.prime) & FNV_MASK_32


# ──────────────────────────────────────────────────────────────────────────
# WeightBitstream — encrypt + integrity-hash a compiled weight blob
# ──────────────────────────────────────────────────────────────────────────


class WeightBitstream:
    """Encrypt and integrity-hash a compiled weight blob.

    Combines :class:`LFSRCipher` (stream encryption) and
    :class:`IntegrityHash` (FNV-1a) into a tamper-evident envelope.

    Envelope binary layout (all little-endian)::

        ┌────────────────────────────────────────────────────────┐
        │ Header (24 bytes)                                      │
        │   magic       : 4s   b"WSEC"                            │
        │   version     : I    1                                  │
        │   hash         : I    FNV-1a 32-bit hash of plaintext   │
        │   payload_len  : I    length of encrypted payload       │
        │   reserved    : 8s   zero padding                       │
        ├────────────────────────────────────────────────────────┤
        │ Encrypted payload (payload_len bytes)                  │
        │   LFSRCipher.encrypt(plaintext_weights)                 │
        └────────────────────────────────────────────────────────┘

    The hash is computed over the *plaintext* weights before encryption.
    On unpack, the payload is decrypted first, then the hash is
    recomputed and compared — a mismatch raises :class:`TamperError`.

    Examples
    --------
    >>> blob = WeightBitstream.pack(b"weight_data", key=0xCAFEBABE)
    >>> WeightBitstream.unpack(blob, key=0xCAFEBABE)
    b'weight_data'
    """

    @staticmethod
    def pack(weights: bytes, key: int) -> bytes:
        """Encrypt and hash *weights*, returning a tamper-evident blob.

        Parameters
        ----------
        weights : bytes
            Plaintext compiled weight data (e.g. output of
            ``SpectralChipCompiler.compile_model``).
        key : int
            32-bit LFSR cipher key.

        Returns
        -------
        bytes
            Encrypted blob with integrity-hash header.
        """
        if not isinstance(weights, (bytes, bytearray)):
            raise TypeError(
                f"weights must be bytes or bytearray, "
                f"got {type(weights).__name__}"
            )
        weights = bytes(weights)

        # Compute integrity hash of the plaintext
        hasher = IntegrityHash()
        digest = hasher.compute(weights)

        # Encrypt the plaintext
        cipher = LFSRCipher(key=key)
        encrypted = cipher.encrypt(weights)

        # Build header
        header = struct.pack(
            WB_HEADER_FORMAT,
            WB_MAGIC,
            WB_VERSION,
            digest,
            len(encrypted),
            b"\x00" * 8,
        )
        return header + encrypted

    @staticmethod
    def unpack(data: bytes, key: int) -> bytes:
        """Decrypt and verify a blob produced by :meth:`pack`.

        Parameters
        ----------
        data : bytes
            Encrypted blob from :meth:`pack`.
        key : int
            32-bit LFSR cipher key (must match the key used for packing).

        Returns
        -------
        bytes
            Recovered plaintext weights.

        Raises
        ------
        TamperError
            If the integrity hash does not match (tamper detected), the
            magic is wrong, or the version is unsupported.
        ValueError
            If the blob is truncated or malformed.
        """
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(
                f"data must be bytes or bytearray, got {type(data).__name__}"
            )
        data = bytes(data)

        if len(data) < WB_HEADER_SIZE:
            raise ValueError(
                f"unpack: blob too short ({len(data)} < {WB_HEADER_SIZE} bytes)"
            )

        magic, version, stored_hash, payload_len, _reserved = struct.unpack(
            WB_HEADER_FORMAT, data[:WB_HEADER_SIZE]
        )

        if magic != WB_MAGIC:
            raise TamperError(
                f"unpack: bad magic {magic!r} (expected {WB_MAGIC!r})"
            )
        if version != WB_VERSION:
            raise TamperError(
                f"unpack: unsupported version {version} "
                f"(expected {WB_VERSION})"
            )
        if len(data) < WB_HEADER_SIZE + payload_len:
            raise TamperError(
                f"unpack: blob truncated (expected {payload_len} bytes of "
                f"payload, got {len(data) - WB_HEADER_SIZE})"
            )

        encrypted = data[WB_HEADER_SIZE : WB_HEADER_SIZE + payload_len]

        # Decrypt
        cipher = LFSRCipher(key=key)
        plaintext = cipher.decrypt(encrypted)

        # Verify integrity
        hasher = IntegrityHash()
        computed_hash = hasher.compute(plaintext)
        if computed_hash != (stored_hash & FNV_MASK_32):
            raise TamperError(
                f"unpack: integrity check failed — hash mismatch "
                f"(expected 0x{stored_hash & FNV_MASK_32:08X}, "
                f"got 0x{computed_hash:08X})"
            )

        return plaintext


# ──────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────


def _self_test() -> None:
    """Quick self-test for manual verification."""
    # LFSRCipher round-trip
    for key in [0xDEADBEEF, 0x12345678, 0x00000001, 0xFFFFFFFF]:
        plaintext = b"The quick brown fox jumps over the lazy dog."
        c = LFSRCipher(key=key)
        ct = c.encrypt(plaintext)
        assert ct != plaintext, f"cipher did not change data for key {key:#x}"
        c2 = LFSRCipher(key=key)
        assert c2.decrypt(ct) == plaintext, "round-trip failed"

    # Bypass mode
    assert LFSRCipher(key=0).encrypt(b"test") == b"test"

    # IntegrityHash
    h = IntegrityHash()
    assert h.compute(b"") == FNV_OFFSET_BASIS_32
    assert h.verify(b"hello", h.compute(b"hello"))

    # WeightBitstream
    weights = b"\x00\x01\x02\x03\xFF\xFE\xFD\xFC" * 10
    blob = WeightBitstream.pack(weights, key=0xCAFEBABE)
    assert WeightBitstream.unpack(blob, key=0xCAFEBABE) == weights

    # Tamper detection
    tampered = bytearray(blob)
    tampered[WB_HEADER_SIZE] ^= 0xFF
    try:
        WeightBitstream.unpack(bytes(tampered), key=0xCAFEBABE)
        raise AssertionError("tamper not detected!")
    except TamperError:
        pass

    print("security.py self-test passed.")


if __name__ == "__main__":
    _self_test()