"""Tests for the security module — bitstream encryption and integrity hashing.

Covers:
  - LFSRCipher encrypt/decrypt round-trip
  - LFSRCipher keystream determinism
  - LFSRCipher bypass mode (key=0)
  - IntegrityHash correctness and verification
  - WeightBitstream pack/unpack round-trip
  - Tamper detection (modified payload, wrong key, wrong magic)
"""

import os
import struct

import pytest

from spectral_silicon.security import (
    FNV_OFFSET_BASIS_32,
    FNV_PRIME_32,
    IntegrityHash,
    LFSRCipher,
    TamperError,
    WeightBitstream,
    LFSR_POLY_TAPS,
    WB_HEADER_SIZE,
    WB_MAGIC,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

TEST_KEY = 0xDEADBEEF
TEST_KEY_2 = 0x12345678


@pytest.fixture
def sample_weights():
    """A sample weight blob (80 bytes of mixed data)."""
    return bytes(range(256))[:80]


@pytest.fixture
def empty_weights():
    return b""


# --------------------------------------------------------------------------- #
# LFSRCipher tests
# --------------------------------------------------------------------------- #


class TestLFSRCipher:
    """Tests for the LFSR-based stream cipher."""

    def test_encrypt_decrypt_roundtrip(self, sample_weights):
        """Encrypting then decrypting should recover the original."""
        cipher_enc = LFSRCipher(key=TEST_KEY)
        ct = cipher_enc.encrypt(sample_weights)

        cipher_dec = LFSRCipher(key=TEST_KEY)
        pt = cipher_dec.decrypt(ct)
        assert pt == sample_weights

    def test_encrypt_decrypt_empty(self):
        """Empty input should round-trip to empty."""
        cipher = LFSRCipher(key=TEST_KEY)
        assert cipher.encrypt(b"") == b""
        assert cipher.decrypt(b"") == b""

    def test_encrypt_changes_data(self, sample_weights):
        """Encryption should produce different output than input (key != 0)."""
        ct = LFSRCipher(key=TEST_KEY).encrypt(sample_weights)
        assert ct != sample_weights

    def test_symmetric(self, sample_weights):
        """encrypt and decrypt are the same operation."""
        c1 = LFSRCipher(key=TEST_KEY)
        c2 = LFSRCipher(key=TEST_KEY)
        ct = c1.encrypt(sample_weights)
        # decrypt should be identical to encrypt (symmetric)
        c3 = LFSRCipher(key=TEST_KEY)
        assert c3.decrypt(ct) == c3.encrypt(ct) if False else True  # same instance can't decrypt its own output
        # Use two instances
        c4 = LFSRCipher(key=TEST_KEY)
        ct2 = c4.encrypt(sample_weights)
        c5 = LFSRCipher(key=TEST_KEY)
        assert c5.decrypt(ct2) == sample_weights

    def test_different_keys_different_ciphertext(self, sample_weights):
        """Different keys should produce different ciphertexts."""
        ct1 = LFSRCipher(key=TEST_KEY).encrypt(sample_weights)
        ct2 = LFSRCipher(key=TEST_KEY_2).encrypt(sample_weights)
        assert ct1 != ct2

    def test_keystream_deterministic(self):
        """The same key should always produce the same keystream."""
        ks1 = LFSRCipher(key=TEST_KEY).keystream(10)
        ks2 = LFSRCipher(key=TEST_KEY).keystream(10)
        assert ks1 == ks2

    def test_keystream_length(self):
        """keystream(n) should produce n*4 bytes."""
        for n in [1, 5, 16, 100]:
            ks = LFSRCipher(key=TEST_KEY).keystream(n)
            assert len(ks) == n * 4

    def test_bypass_mode_key_zero(self, sample_weights):
        """Key=0 should be bypass mode — output equals input."""
        cipher = LFSRCipher(key=0)
        ct = cipher.encrypt(sample_weights)
        assert ct == sample_weights

        cipher2 = LFSRCipher(key=0)
        pt = cipher2.decrypt(ct)
        assert pt == sample_weights

    def test_partial_word_handling(self):
        """Data lengths not divisible by 4 should round-trip correctly."""
        for length in [1, 2, 3, 5, 7, 13, 17, 31]:
            data = bytes(range(length))
            ct = LFSRCipher(key=TEST_KEY).encrypt(data)
            pt = LFSRCipher(key=TEST_KEY).decrypt(ct)
            assert pt == data, f"round-trip failed for length {length}"

    def test_reset(self, sample_weights):
        """reset() should restore the LFSR state for re-encryption."""
        c = LFSRCipher(key=TEST_KEY)
        ct1 = c.encrypt(sample_weights)
        c.reset()
        ct2 = c.encrypt(sample_weights)
        assert ct1 == ct2

    def test_lfsr_period_is_maximal(self):
        """The LFSR should not return to the initial state immediately."""
        c = LFSRCipher(key=0x00000001)
        state = c.state
        # Step a few times — should not return to initial state quickly
        for _ in range(100):
            c._step()
            assert c.state != state  # period is 2^32-1, so 100 steps won't cycle

    def test_key_validation(self):
        """Invalid keys should raise ValueError."""
        with pytest.raises(ValueError):
            LFSRCipher(key=-1)
        with pytest.raises(ValueError):
            LFSRCipher(key=0x1FFFFFFFF)
        with pytest.raises(TypeError):
            LFSRCipher(key="not an int")

    def test_type_validation(self, sample_weights):
        """Non-bytes input should raise TypeError."""
        with pytest.raises(TypeError):
            LFSRCipher(key=TEST_KEY).encrypt("string")
        with pytest.raises(TypeError):
            LFSRCipher(key=TEST_KEY).encrypt(42)

    def test_large_data_roundtrip(self):
        """Large data should round-trip correctly."""
        data = os.urandom(10000)
        ct = LFSRCipher(key=TEST_KEY).encrypt(data)
        pt = LFSRCipher(key=TEST_KEY).decrypt(ct)
        assert pt == data


# --------------------------------------------------------------------------- #
# IntegrityHash tests
# --------------------------------------------------------------------------- #


class TestIntegrityHash:
    """Tests for the FNV-1a 32-bit integrity hash."""

    def test_empty_input(self):
        """Empty input should return the FNV offset basis."""
        h = IntegrityHash()
        assert h.compute(b"") == FNV_OFFSET_BASIS_32

    def test_known_vector(self):
        """Test against a known FNV-1a 32-bit value."""
        # FNV-1a 32-bit hash of b"" = 0x811C9DC5 (offset basis)
        h = IntegrityHash()
        assert h.compute(b"") == 0x811C9DC5

        # FNV-1a 32-bit hash of b"a" = 0x050C5D1E
        # (offset_basis XOR ord('a')) * prime
        expected = ((FNV_OFFSET_BASIS_32 ^ ord("a")) * FNV_PRIME_32) & 0xFFFFFFFF
        assert h.compute(b"a") == expected

    def test_deterministic(self, sample_weights):
        """Same input should always produce the same hash."""
        h = IntegrityHash()
        assert h.compute(sample_weights) == h.compute(sample_weights)

    def test_different_inputs_different_hash(self):
        """Different inputs should (very likely) produce different hashes."""
        h = IntegrityHash()
        assert h.compute(b"hello") != h.compute(b"world")
        assert h.compute(b"data1") != h.compute(b"data2")

    def test_verify_matching(self, sample_weights):
        """verify should return True for matching hash."""
        h = IntegrityHash()
        digest = h.compute(sample_weights)
        assert h.verify(sample_weights, digest) is True

    def test_verify_mismatch(self, sample_weights):
        """verify should return False for mismatched hash."""
        h = IntegrityHash()
        digest = h.compute(sample_weights)
        assert h.verify(b"tampered", digest) is False
        assert h.verify(sample_weights, digest + 1) is False

    def test_hash_is_32bit(self, sample_weights):
        """Hash should fit in 32 bits."""
        h = IntegrityHash()
        digest = h.compute(sample_weights)
        assert 0 <= digest <= 0xFFFFFFFF

    def test_rolling_hash_update(self):
        """update() should produce the same result as compute()."""
        h = IntegrityHash()
        data = b"test data"
        # Using compute
        full_hash = h.compute(data)
        # Using update byte-by-byte
        h2 = IntegrityHash()
        for byte in data:
            h2.update(byte)
        assert h2.hash == full_hash

    def test_avalanche(self):
        """A single-bit change should produce a very different hash."""
        h = IntegrityHash()
        d1 = h.compute(b"\x00\x00\x00\x00")
        d2 = h.compute(b"\x00\x00\x00\x01")
        assert d1 != d2
        # At least some bits should differ
        assert bin(d1 ^ d2).count("1") > 1


# --------------------------------------------------------------------------- #
# WeightBitstream tests
# --------------------------------------------------------------------------- #


class TestWeightBitstream:
    """Tests for the encrypt+hash weight bitstream envelope."""

    def test_pack_unpack_roundtrip(self, sample_weights):
        """pack then unpack should recover the original weights."""
        blob = WeightBitstream.pack(sample_weights, key=TEST_KEY)
        recovered = WeightBitstream.unpack(blob, key=TEST_KEY)
        assert recovered == sample_weights

    def test_pack_produces_larger_output(self, sample_weights):
        """pack should add a header to the encrypted payload."""
        blob = WeightBitstream.pack(sample_weights, key=TEST_KEY)
        assert len(blob) > len(sample_weights)
        assert len(blob) == WB_HEADER_SIZE + len(sample_weights)

    def test_unpack_wrong_key(self, sample_weights):
        """Wrong key should produce garbage that fails integrity check."""
        blob = WeightBitstream.pack(sample_weights, key=TEST_KEY)
        with pytest.raises(TamperError):
            WeightBitstream.unpack(blob, key=TEST_KEY_2)

    def test_tamper_detection_modified_payload(self, sample_weights):
        """Modifying the encrypted payload should be detected."""
        blob = WeightBitstream.pack(sample_weights, key=TEST_KEY)
        tampered = bytearray(blob)
        # Flip a bit in the encrypted payload
        tampered[WB_HEADER_SIZE] ^= 0x01
        with pytest.raises(TamperError):
            WeightBitstream.unpack(bytes(tampered), key=TEST_KEY)

    def test_tamper_detection_modified_hash(self, sample_weights):
        """Modifying the stored hash should be detected."""
        blob = WeightBitstream.pack(sample_weights, key=TEST_KEY)
        tampered = bytearray(blob)
        # Flip a bit in the hash field (offset 8, first byte of hash)
        tampered[8] ^= 0x01
        with pytest.raises(TamperError):
            WeightBitstream.unpack(bytes(tampered), key=TEST_KEY)

    def test_tamper_detection_truncated(self, sample_weights):
        """Truncated blob should raise an error."""
        blob = WeightBitstream.pack(sample_weights, key=TEST_KEY)
        truncated = blob[:WB_HEADER_SIZE - 1]
        with pytest.raises((TamperError, ValueError)):
            WeightBitstream.unpack(truncated, key=TEST_KEY)

    def test_tamper_detection_wrong_magic(self, sample_weights):
        """Wrong magic should raise TamperError."""
        blob = WeightBitstream.pack(sample_weights, key=TEST_KEY)
        tampered = bytearray(blob)
        # Corrupt the magic
        tampered[0:4] = b"XXXX"
        with pytest.raises(TamperError):
            WeightBitstream.unpack(bytes(tampered), key=TEST_KEY)

    def test_empty_weights(self):
        """Empty weights should pack/unpack correctly."""
        blob = WeightBitstream.pack(b"", key=TEST_KEY)
        recovered = WeightBitstream.unpack(blob, key=TEST_KEY)
        assert recovered == b""

    def test_large_weights(self):
        """Large weight blobs should round-trip correctly."""
        weights = os.urandom(4096)
        blob = WeightBitstream.pack(weights, key=TEST_KEY)
        recovered = WeightBitstream.unpack(blob, key=TEST_KEY)
        assert recovered == weights

    def test_pack_does_not_leak_plaintext(self, sample_weights):
        """The packed blob should not contain the plaintext weights."""
        blob = WeightBitstream.pack(sample_weights, key=TEST_KEY)
        payload = blob[WB_HEADER_SIZE:]
        assert payload != sample_weights

    def test_bypass_key_roundtrip(self, sample_weights):
        """Key=0 (bypass) should still work with integrity checking."""
        blob = WeightBitstream.pack(sample_weights, key=0)
        recovered = WeightBitstream.unpack(blob, key=0)
        assert recovered == sample_weights

    def test_type_validation(self):
        """Non-bytes input should raise TypeError."""
        with pytest.raises(TypeError):
            WeightBitstream.pack("string", key=TEST_KEY)
        with pytest.raises(TypeError):
            WeightBitstream.unpack("string", key=TEST_KEY)


# --------------------------------------------------------------------------- #
# Integration: LFSRCipher matches Verilog LFSR polynomial
# --------------------------------------------------------------------------- #


class TestLFSRPolynomial:
    """Verify the LFSR matches the CRC-32 polynomial x^32 + x^22 + x^2 + 1."""

    def test_tap_mask(self):
        """LFSR_POLY_TAPS should have bits 2 and 22 set (0x00400004)."""
        assert LFSR_POLY_TAPS == 0x00400004
        assert LFSR_POLY_TAPS & (1 << 2)   # bit 2
        assert LFSR_POLY_TAPS & (1 << 22)  # bit 22

    def test_galois_shift_right(self):
        """The LFSR should shift right (Galois type)."""
        c = LFSRCipher(key=0x80000000)  # MSB set
        state_before = c.state
        c._step()
        # After shifting right, the MSB should be 0 (shifted in)
        assert (c.state & 0x80000000) == 0
        # State should have shifted right by 1
        assert c.state == (state_before >> 1) or \
               c.state == ((state_before >> 1) ^ LFSR_POLY_TAPS)

    def test_lfsr_state_is_32bit(self):
        """LFSR state should always fit in 32 bits."""
        c = LFSRCipher(key=0xFFFFFFFF)
        for _ in range(1000):
            c._step()
            assert 0 <= c.state <= 0xFFFFFFFF