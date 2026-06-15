"""Persistence support helpers."""

import hashlib


def get_md5_hexdigest(content):
    """Return the MD5 hexdigest of an object string representation."""
    return hashlib.md5(str(content).encode()).hexdigest()


def exclude_unshared_keys_and_identical_values(list_of_dicts):
    """Filter dicts to exclude unshared keys and keys with identical values."""
    if not list_of_dicts:
        return []

    common_keys = set(list_of_dicts[0].keys())
    for d in list_of_dicts[1:]:
        common_keys.intersection_update(d.keys())

    keys_to_exclude = {
        key
        for key in common_keys
        if all(d[key] == list_of_dicts[0][key] for d in list_of_dicts[1:])
    }

    return [
        {k: v for k, v in original_dict.items() if k not in keys_to_exclude}
        for original_dict in list_of_dicts
    ]
