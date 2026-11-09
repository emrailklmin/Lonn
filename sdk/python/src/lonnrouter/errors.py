"""Stable exceptions for application error handling."""


class LonnError(Exception):
    """Base class for SDK request and response errors."""


class LonnAPIError(LonnError):
    """The server returned a non-success HTTP status."""

    def __init__(self, status_code: int):
        self.status_code = status_code
        # Avoid including validation bodies, which may echo private prompts.
        super().__init__(f"Lonn API returned HTTP {status_code}")


class LonnConnectionError(LonnError):
    """The client could not communicate with the server."""


class LonnTimeoutError(LonnConnectionError):
    """The HTTP request timed out; server-side work may still be running."""


class LonnResponseError(LonnError):
    """The server returned an invalid response for this SDK contract."""
