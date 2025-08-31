from google.genai import Client
from google.genai.types import GenerateContentResponse
from loguru import logger
from pydantic import BaseModel

from src.rate_limit.token_bucket import AsyncTokenBucket, TokenBucket


class Curpos(BaseModel):
    text: str
    score: int


class QueryBier(BaseModel):
    curpos: list[Curpos]


class GeminiClient:
    def __init__(self, api_key: str, limiter: AsyncTokenBucket):
        self.client = Client(api_key=api_key)
        self.limiter = limiter
        self.key = api_key

    async def generate_text(self, prompt: str) -> QueryBier | None:
        await self.limiter.acquire()
        logger.info("getting res for key {}", self.key)
        try:
            response: GenerateContentResponse = (
                await self.client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": QueryBier,
                    },
                )
            )
            return response.parsed
        except Exception as e:
            logger.error(e)
            return None
