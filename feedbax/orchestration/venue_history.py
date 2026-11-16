"""Read the controller's authenticated Host journal without attaching to the venue."""

import json
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

from feedbax.host.protocol import Event, JobsSnapshot
from feedbax.orchestration.session import SessionDriver, SessionError


class VenueHistory:
    def __init__(self, driver: SessionDriver) -> None:
        self.driver = driver
        with driver.connection() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS session_observations(
                    session TEXT PRIMARY KEY, stream TEXT NOT NULL, cursor INTEGER NOT NULL, snapshot TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS session_events(
                    session TEXT NOT NULL, cursor INTEGER NOT NULL, event TEXT NOT NULL,
                    PRIMARY KEY(session, cursor)
                );
                """
            )

    def observation(self, session: UUID | str) -> tuple[str, int, JobsSnapshot] | None:
        with self.driver.connection() as db:
            row = db.execute(
                "SELECT stream, cursor, snapshot FROM session_observations WHERE session=?", (str(session),)
            ).fetchone()
        return (row[0], row[1], json.loads(row[2])) if row is not None else None

    def cursors(self) -> dict[UUID, tuple[str, int]]:
        with self.driver.connection() as db:
            rows = db.execute("SELECT session, stream, cursor FROM session_observations").fetchall()
        return {UUID(session): (stream, cursor) for session, stream, cursor in rows}

    def destination_stores(self, session: UUID | str) -> dict[str, Path]:
        with self.driver.connection() as db:
            destinations = db.execute(
                """SELECT DISTINCT json_extract(request, '$.project'), json_extract(request, '$.local_store')
                   FROM session_delegations WHERE session=? AND json_extract(request, '$.local_store') IS NOT NULL""",
                (str(session),),
            ).fetchall()
        stores = dict(destinations)
        if len(stores) != len(destinations):
            raise SessionError("The machine has more than one destination Store for the same project.")
        return {project: Path(store) for project, store in stores.items()}

    def log_tails(self, session: UUID | str) -> dict[str, list[str]]:
        with self.driver.connection() as db:
            logs = db.execute(
                """SELECT json_extract(event, '$.job_id'), substr(json_extract(event, '$.data.line'), 1, 1024)
                   FROM session_events WHERE session=? AND json_extract(event, '$.kind')='log'
                   ORDER BY cursor DESC LIMIT 200""", (str(session),),
            ).fetchall()
        tails: dict[str, list[str]] = {}
        for job, line in logs:
            tail = tails.setdefault(job, [])
            if isinstance(line, str) and len(tail) < 12:
                tail.append(line)
        for tail in tails.values():
            tail.reverse()
        return tails

    def events(
        self, session: UUID, stream: str, *, after: int, through: int, projects: Sequence[str],
    ) -> tuple[int, tuple[Event, ...]]:
        if not projects:
            return through, ()
        placeholders = ", ".join("?" for _ in projects)
        with self.driver.connection() as db:
            db.execute("BEGIN")
            row = db.execute(
                "SELECT stream, cursor FROM session_observations WHERE session=?", (str(session),)
            ).fetchone()
            if row is None or row[0] != stream or not 0 <= after <= through <= row[1]:
                raise SessionError("The machine's retained event boundary changed.")
            events = []
            remaining = 8 * 1024 * 1024
            for (content,) in db.execute(
                f"""SELECT event FROM session_events
                    WHERE session=? AND cursor>? AND cursor<=? AND json_extract(event,'$.project') IN ({placeholders})
                    ORDER BY cursor LIMIT 257""",
                (str(session), after, through, *projects),
            ):
                size = len(content.encode())
                if len(events) == 256 or events and size > remaining:
                    return events[-1]["cursor"], tuple(events)
                remaining -= size
                events.append(json.loads(content))
            return through, tuple(events)

    def record(
        self, session: str, observed: tuple[str, int, JobsSnapshot] | None, stream: str,
        cursor: int, snapshot: JobsSnapshot, events: Sequence[Event],
    ) -> None:
        if snapshot["cursor"] < cursor or (observed is not None and snapshot["cursor"] < observed[2]["cursor"]):
            raise SessionError("The venue Host event stream no longer matches the saved cursor.")
        with self.driver.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                "SELECT stream, cursor, snapshot FROM session_observations WHERE session=?", (session,)
            ).fetchone()
            current = (current[0], current[1], json.loads(current[2])) if current is not None else None
            if current != observed:
                raise SessionError("Another poll advanced the session observation. Poll again.")
            db.executemany(
                "INSERT INTO session_events VALUES (?, ?, ?)",
                ((session, item["cursor"], json.dumps(item)) for item in events),
            )
            db.execute(
                """INSERT INTO session_observations VALUES (?, ?, ?, ?)
                   ON CONFLICT(session) DO UPDATE SET cursor=excluded.cursor, snapshot=excluded.snapshot""",
                (session, stream, cursor, json.dumps(snapshot)),
            )
