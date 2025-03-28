from fastapi import APIRouter, Request
from .views import chat, Chat  # Import the Chat class

chat_router = APIRouter()

@chat_router.post("/stream")
async def stream_chat(request: Request):
    # Getting the input query from the request
    data = await request.json()
    query = data.get("query", "")
    
    # Create a Chat object and pass it to the chat function
    chat_model = Chat(query=query)
    
    # Returning the chat response as a streaming response
    return await chat(chat_model)
