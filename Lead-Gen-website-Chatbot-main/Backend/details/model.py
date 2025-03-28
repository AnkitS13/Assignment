from pydantic import BaseModel

class Details(BaseModel):
    name: str
    email: str
    company: str
    requirements: str