from fastapi import APIRouter
from .views import get_details

leads_router = APIRouter()

leads_router.post("")(get_details)