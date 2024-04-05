
from functools import partial
import os
import sys


# TODO: Document this environment variable
tqdm_mode = os.environ.get("FEEDBAX_TQDM", "auto")


match tqdm_mode:
    # TODO: This leads to a recursion issue, probably with tqdm.write and the logger
    # case "rich":
    #     from tqdm.rich import tqdm
    case "notebook":
        from tqdm.notebook import tqdm as _tqdm
    case "cli" | "console":
        from tqdm import tqdm as _tqdm
        _tqdm_write = partial(_tqdm.write, file=sys.stderr, end="")
    case "auto" | _:
        from tqdm.auto import tqdm as _tqdm
        _tqdm_write = partial(_tqdm.write, file=sys.stdout, end="")