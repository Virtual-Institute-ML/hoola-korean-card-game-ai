from run_game import resolve_seed


def test_resolve_seed_preserves_explicit_seed():
    assert resolve_seed(12345) == 12345


def test_resolve_seed_generates_valid_integer_when_omitted():
    seed = resolve_seed(None)
    assert isinstance(seed, int)
    assert 0 <= seed < 2**63


def test_resolve_seed_usually_changes_between_calls():
    assert resolve_seed(None) != resolve_seed(None)
