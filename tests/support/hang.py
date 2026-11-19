"""The one limit after which a test, or a gate it installs in a worker, treats a wait as hung.

A wait in the tests ends when the state it waits for arrives. This limit only turns a hang into a failure;
it is not an estimate of how long anything takes, so no wait carries a deadline of its own.
"""

HANG_SECONDS = 120
