import difflib
import textwrap
from itertools import zip_longest

import equinox as eqx


def highlight_string_diff(obj1, obj2):
    """Given two objects, give a string that highlights the differences in
    their string representations.

    This can be useful for identifying slight differences in large PyTrees.

    Source: https://stackoverflow.com/a/76946768
    """
    str1 = repr(obj1)
    str2 = repr(obj2)

    matcher = difflib.SequenceMatcher(None, str1, str2)

    str2_new = ""
    i = 0
    for m in matcher.get_matching_blocks():
        if m.b > i:
            str2_new += str2[i : m.b]
        str2_new += f"\033[91m{str2[m.b : m.b + m.size]}\033[0m"
        i = m.b + m.size

    return str2_new.replace("\\n", "\n")


def print_trees_side_by_side(tree1, tree2, column_width=60, separator="|"):
    """Given two PyTrees, print their pretty representations side-by-side."""
    strs1 = eqx.tree_pformat(tree1).split("\n")
    strs2 = eqx.tree_pformat(tree2).split("\n")

    def wrap_text(text, width):
        return textwrap.wrap(text, width) or [""]

    wrapped1 = [wrap_text(s, column_width) for s in strs1]
    wrapped2 = [wrap_text(s, column_width) for s in strs2]

    for w1, w2 in zip_longest(wrapped1, wrapped2, fillvalue=[""]):
        max_lines = max(len(w1), len(w2))

        for i in range(max_lines):
            line1 = w1[i] if i < len(w1) else ""
            line2 = w2[i] if i < len(w2) else ""
            print(f"{line1:<{column_width}} {separator} {line2:<{column_width}}")


# def _simple_module_pprint(name, *children, **kwargs):
#     return bracketed(
#         pp.text(name),
#         kwargs['indent'],
#         [tree_pp(child, **kwargs) for child in children],
#         '(',
#         ')'
#     )
