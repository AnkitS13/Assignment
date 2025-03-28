from fastapi import FastAPI
from chat.routes import chat_router as chat
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Lead Generation Chatbot - Backend API")

app.include_router(chat, prefix="/chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)