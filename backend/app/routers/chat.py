from fastapi import APIRouter
from pydantic import BaseModel
from ..llm import chat_with_llm

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    print(f"Received chat message: {req.message}")
    answer = chat_with_llm(req.message)
    return {"answer": answer}
