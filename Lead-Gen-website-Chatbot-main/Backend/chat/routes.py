from fastapi import APIRouter
from .views import chat

chat_router = APIRouter()

chat_router.post("")(chat)