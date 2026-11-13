"""Same-origin browser transport bridged through the private native control socket."""

from __future__ import annotations

import hmac
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool
from starlette.requests import HTTPConnection
from starlette.responses import HTMLResponse, JSONResponse, Response

from feedbax.contracts.base import StrictModel
from feedbax.desktop.control import CONTROL_ROUTE, Endpoint
from feedbax.web.services.project_folder import ProjectFolderNotFound

_COOKIE = "feedbax_studio"
_CONNECT = CONTROL_ROUTE + "/connect"
_EXCHANGE = CONTROL_ROUTE + "/exchange"
_BOOTSTRAP = """<!doctype html>
<meta charset="utf-8"><title>Feedbax Studio</title>
<p id="status">Connecting to Feedbax Studio…</p>
<script>
const ticket = location.hash.slice(1);
history.replaceState(null, "", location.pathname);
fetch("/_desktop/exchange", { method: "POST", headers: {"X-Feedbax-Ticket": ticket} })
  .then(response => {
    if (!response.ok) throw Error();
    const project = response.headers.get("X-Feedbax-Project");
    if (project) location.replace("/?project=" + encodeURIComponent(project));
    else if (response.headers.get("X-Feedbax-Activity") === "1") location.replace("/?activity=1");
    else throw Error();
  })
  .catch(() => {
    document.querySelector("#status").textContent =
      "Could not connect. Select Reconnect in the Feedbax Studio menu.";
  });
</script>
"""
_BROWSER_HEADERS = {
    "cache-control": "no-store",
    "referrer-policy": "no-referrer",
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
}


class _TicketRequest(StrictModel):
    project_folder: Path | None = None
    project_file: Path | None = None
    project_id: str | None = None
    folder_identity: tuple[int, int] | None = None
    activity: bool = False


@dataclass(frozen=True)
class _Ticket:
    expires: float
    project_id: str | None


class DesktopTransport:
    def __init__(self, app, endpoint: Endpoint) -> None:
        self.app = app
        self.endpoint = endpoint
        self.host = f"127.0.0.1:{endpoint.port}"
        self.session = secrets.token_urlsafe(32)
        self.tickets: dict[str, _Ticket] = {}

    def issue_ticket(self, project_id: str | None) -> str:
        now = time.monotonic()
        self.tickets = {token: ticket for token, ticket in self.tickets.items() if ticket.expires > now}
        if len(self.tickets) >= 32:
            del self.tickets[next(iter(self.tickets))]
        token = secrets.token_urlsafe(32)
        self.tickets[token] = _Ticket(now + 20, project_id)
        return token

    def register_ticket_route(self, is_stopping: Callable[[], bool]) -> None:
        @self.app.post(CONTROL_ROUTE + "/ticket")
        async def ticket(request: _TicketRequest):
            if is_stopping():
                raise HTTPException(503, "Studio service is stopping")
            if sum((
                request.project_folder is not None, request.project_file is not None,
                request.project_id is not None, request.activity,
            )) > 1:
                raise HTTPException(422, "Select one folder, declaration, project, or Activity.")
            if request.folder_identity is not None and request.project_folder is None:
                raise HTTPException(422, "Restoring a project requires its folder.")
            if request.activity:
                return {"ticket": self.issue_ticket(None)}
            folders = self.app.state.project_folders
            try:
                if request.project_folder is not None:
                    document, location = await run_in_threadpool(
                        folders.open_native_folder, request.project_folder, request.folder_identity,
                    )
                elif request.project_file is not None:
                    document, location = await run_in_threadpool(folders.open_native_document, request.project_file)
                elif request.project_id is not None:
                    document, location = await run_in_threadpool(folders.open_native_id, request.project_id)
                else:
                    document, location = await run_in_threadpool(folders.new_native_project)
            except ProjectFolderNotFound as error:
                raise HTTPException(404, "The selected Studio project folder is missing.") from error
            except (OSError, ValueError) as error:
                raise HTTPException(422, str(error)) from error
            if is_stopping():
                raise HTTPException(503, "Studio service is stopping")
            project_id = document.snapshot.project_id
            return {
                "ticket": self.issue_ticket(project_id),
                "project": {"id": project_id, "title": document.configuration.project},
                "folder": {"path": str(location.path), "identity": location.identity},
            }

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return

        connection = HTTPConnection(scope)
        host = connection.headers.get("host")
        origin = connection.headers.get("origin")
        path = scope["path"]
        is_browser = scope["server"] == ("127.0.0.1", self.endpoint.port)
        if scope["server"] is None and host == "localhost" and origin is None:
            if scope["type"] == "http" and path.startswith(f"{CONTROL_ROUTE}/"):
                await self.app(scope, receive, send)
                return
        if not is_browser or host != self.host or origin not in (None, self.endpoint.origin):
            await self._deny(scope, receive, send)
            return

        if scope["type"] == "websocket":
            if origin == self.endpoint.origin and self._authenticated(connection):
                await self.app(scope, receive, send)
            else:
                await self._deny(scope, receive, send)
            return

        method = scope["method"]
        if path == _CONNECT and method == "GET":
            response = HTMLResponse(
                _BOOTSTRAP, headers={
                    **_BROWSER_HEADERS,
                    "content-security-policy": (
                        "default-src 'none'; script-src 'unsafe-inline'; connect-src 'self'"
                    ),
                },
            )
        elif path == _EXCHANGE and method == "POST" and origin == self.endpoint.origin:
            token = connection.headers.get("x-feedbax-ticket", "")
            ticket = self.tickets.pop(token, None)
            if ticket is None or ticket.expires <= time.monotonic():
                await self._deny(scope, receive, send)
                return
            response = Response(status_code=204, headers=_BROWSER_HEADERS)
            if ticket.project_id is None:
                response.headers["x-feedbax-activity"] = "1"
            else:
                response.headers["x-feedbax-project"] = ticket.project_id
            response.set_cookie(_COOKIE, self.session, httponly=True, samesite="strict")
        elif (
            path.startswith(f"{CONTROL_ROUTE}/")
            or not self._authenticated(connection)
            or (method not in ("GET", "HEAD", "OPTIONS") and origin != self.endpoint.origin)
        ):
            await self._deny(scope, receive, send)
            return
        else:
            async def browser_send(message) -> None:
                if message["type"] == "http.response.start":
                    headers = list(message["headers"])
                    headers = [(key, value) for key, value in headers
                               if key.decode("ascii") not in _BROWSER_HEADERS]
                    headers.extend((key.encode(), value.encode())
                                   for key, value in _BROWSER_HEADERS.items())
                    message = {**message, "headers": headers}
                await send(message)

            await self.app(scope, receive, browser_send)
            return
        await response(scope, receive, send)

    def _authenticated(self, connection: HTTPConnection) -> bool:
        cookie = connection.cookies.get(_COOKIE, "")
        return cookie.isascii() and hmac.compare_digest(cookie, self.session)

    @staticmethod
    async def _deny(scope, receive, send) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
        else:
            await JSONResponse({"detail": "desktop service authorization required"},
                               status_code=403, headers=_BROWSER_HEADERS)(scope, receive, send)
