from datetime import datetime
from typing import Annotated, Optional
from beanie import Document, Indexed
from pydantic import EmailStr, Field

class User(Document):
    email: Annotated[EmailStr, Indexed(unique=True)]
    hashed_password: str
    full_name: Optional[str] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "users"

    def __repr__(self) -> str:
        return f"<User {self.email}>"

    def __str__(self) -> str:
        return self.email
