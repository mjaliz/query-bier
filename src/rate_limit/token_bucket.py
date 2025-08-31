import asyncio
import time

from loguru import logger


class TokenBucket:
    def __init__(self, rate=0.16, capacity=5):
        # Initialize the bucket with a rate and capacity
        self.capacity = capacity  # Maximum number of tokens in the bucket
        self._tokens = capacity  # Current number of tokens
        self.rate = rate  # Rate of token addition per second
        self.last_added = time.time()  # Timestamp of last token addition

    def _add_tokens(self):
        # Add tokens to the bucket based on the elapsed time and rate
        now = time.time()
        tokens_to_add = (now - self.last_added) * self.rate
        if tokens_to_add > 0:
            self._tokens = min(self.capacity, self._tokens + tokens_to_add)
            self.last_added = now

    def allow_request(self, num_tokens=1):
        # Check if a request can be allowed based on available tokens
        self._add_tokens()
        if self._tokens >= num_tokens:
            self._tokens -= num_tokens
            return True
        return False


class AsyncTokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.monotonic()

    def _refill(self):
        """Refill the bucket with new tokens based on the elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.last_refill = now
        new_tokens = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)

    async def acquire(self):
        """Acquire a token from the bucket, waiting if necessary."""
        while True:
            self._refill()
            if self.tokens >= 1:
                self.tokens -= 1
                return
            # Calculate the time to wait for the next token and sleep.
            logger.info("sleep for {} seconds", 1 / self.rate)
            await asyncio.sleep(1 / self.rate)
