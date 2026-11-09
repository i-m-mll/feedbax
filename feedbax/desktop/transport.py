"""Same-origin browser transport bridged through the private native control socket."""

from __future__ import annotations

import hmac
import secrets
import time

from starlette.requests import HTTPConnection
from starlette.responses import HTMLResponse, JSONResponse, Response

from feedbax.desktop.control import CONTROL_ROUTE, Endpoint

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
    location.replace("/");
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


class DesktopTransport:
    def __init__(self, app, endpoint: Endpoint) -> None:
        self.app = app
        self.endpoint = endpoint
        self.host = f"127.0.0.1:{endpoint.port}"
        self.session = secrets.token_urlsafe(32)
        self.tickets: dict[str, float] = {}

    def issue_ticket(self) -> str:
        now = time.monotonic()
        self.tickets = {ticket: expiration for ticket, expiration in self.tickets.items()
                        if expiration > now}
        if len(self.tickets) >= 32:
            del self.tickets[next(iter(self.tickets))]
        ticket = secrets.token_urlsafe(32)
        self.tickets[ticket] = now + 20
        return ticket

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
            ticket = connection.headers.get("x-feedbax-ticket", "")
            expiration = self.tickets.pop(ticket, 0)
            if expiration <= time.monotonic():
                await self._deny(scope, receive, send)
                return
            response = Response(status_code=204, headers=_BROWSER_HEADERS)
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
