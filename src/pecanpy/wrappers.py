"""Wrappers used by pecanpy."""
import time


class Timer:
    """Timer for logging runtime of function."""

    def __init__(self, name, verbose=True):
        """Initialize timer wrapper."""
        self.name = name
        self.verbose = verbose

    def __call__(self, func):
        """Call timer decorator."""

        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start

            hrs = int(duration // 3600)
            mins = int(duration % 3600 // 60)
            secs = duration % 60
            print(f"Took {hrs:02d}:{mins:02d}:{secs:05.2f} to {self.name}")

            return result

        return wrapper if self.verbose else func
