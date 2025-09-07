from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class Corpus(SQLModel):
    corpus_id: str
    corpus_text: str
    corpus_query_id: str
    corpus_query_text: str
    qrel_score: int


class Query(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    query_id: str = Field(index=True)
    query_text: str
    corpuses: list[Corpus] = Field(default_factory=list, sa_type=JSONB)
