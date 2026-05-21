from feedbax._tree import allf, anyf, notf


def test_anyf_returns_true_when_any_predicate_matches():
    is_even_or_negative = anyf(lambda x: x % 2 == 0, lambda x: x < 0)

    assert is_even_or_negative(4)
    assert is_even_or_negative(-3)
    assert not is_even_or_negative(3)


def test_allf_returns_true_only_when_all_predicates_match():
    is_even_and_positive = allf(lambda x: x % 2 == 0, lambda x: x > 0)

    assert is_even_and_positive(4)
    assert not is_even_and_positive(-4)
    assert not is_even_and_positive(3)


def test_notf_negates_predicate():
    is_odd = notf(lambda x: x % 2 == 0)

    assert is_odd(3)
    assert not is_odd(4)
