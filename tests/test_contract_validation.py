import pytest

from feedbax.contracts.validation import is_sha256_digest, validate_sha256
from feedbax.contracts.spec_storage import validate_sha256 as storage_validate_sha256


@pytest.mark.parametrize(
    "value", [None, 1, b"a" * 64, "a" * 63, "a" * 65, "A" * 64, "g" * 64]
)
def test_sha256_digest_rejects_noncanonical_values(value: object) -> None:
    assert not is_sha256_digest(value)
    with pytest.raises(ValueError, match="identity must be a lowercase 64-hex sha256 digest"):
        validate_sha256(value, field_name="identity")


def test_sha256_digest_accepts_and_returns_lowercase_hex() -> None:
    digest = "0123456789abcdef" * 4

    assert is_sha256_digest(digest)
    assert validate_sha256(digest, field_name="identity") == digest
    assert storage_validate_sha256(digest, field_name="identity") == digest
