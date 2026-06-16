"""
Base class for middleware components that process data during fitting.
This is a common module that cannot use `common` imports to avoid circular
dependencies.
"""

from abc import ABC, abstractmethod

from .model_data import ModelData


class Middleware(ABC):
    """Base class for middleware components that process data during fitting."""

    def __init__(self):
        self._next_middleware = None

    @abstractmethod
    def process(self, data: ModelData) -> ModelData:
        """Process the data and return the modified data."""
        pass

    def exec(self, data: ModelData) -> ModelData:
        """Execute the middleware processing on the provided data."""
        data = self.process(data)
        if self._next_middleware is not None:
            data = self._next_middleware.exec(data)
        return data

    def next(self, middleware: "Middleware") -> "Middleware":
        """Set the next middleware in the chain."""
        self._next_middleware = middleware
        return middleware


def stack(*middlewares: Middleware) -> Middleware:
    """Utility function to stack multiple middleware components together."""
    if not middlewares:
        raise ValueError("At least one middleware must be provided.")

    for i in range(len(middlewares) - 1):
        middlewares[i].next(middlewares[i + 1])

    return middlewares[0]


class MiddlewareStack(Middleware):
    """Container for chaining multiple middleware components."""

    def __init__(self) -> None:
        self.middlewares = []

    def process(self, data: ModelData) -> ModelData:
        return self.middlewares[0].exec(data) if self.middlewares else data

    def add(self, middleware: Middleware) -> None:
        """Add a middleware to the stack."""
        if self.middlewares:
            self.middlewares[-1].next(middleware)
        self.middlewares.append(middleware)
        
    def clear(self) -> None:
        """Clear all middlewares from the stack."""
        self.middlewares.clear()

    def exec(self, data: ModelData) -> ModelData:
        """Execute the middleware stack on the provided data."""
        if not self.middlewares:
            return data
        return self.process(data)
