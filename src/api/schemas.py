from pydantic import BaseModel


class QueryResponse(BaseModel):
    session_id: str
    final_answer: str
    agents_used: list[str] = []
    latency_ms: float | None = None
    context: dict | None = None


class MessageItem(BaseModel):
    role: str             # "human" | "ai"
    content: str


class SessionResponse(BaseModel):
    session_id: str
    intent: str | None = None
    final_answer: str | None = None
    messages: list[MessageItem] = []
