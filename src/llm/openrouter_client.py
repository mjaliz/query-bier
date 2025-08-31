from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel

from src.rate_limit.token_bucket import AsyncTokenBucket


class Curpos(BaseModel):
    text: str
    score: int


class QueryBier(BaseModel):
    curpos: list[Curpos]


class OpenrouterClient:
    def __init__(self, base_url: str, api_key: str, limiter: AsyncTokenBucket):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.limiter = limiter
        self.key = api_key

    async def generate_text(self, prompt: str) -> QueryBier | None:
        await self.limiter.acquire()
        try:
            response = await self.client.beta.chat.completions.parse(
                model="google/gemini-2.5-flash",
                messages=[{"role": "user", "content": prompt}],
                response_format=QueryBier,
            )

            return response.choices[0].message.parsed
        except Exception as e:
            logger.error(e)
            return None
