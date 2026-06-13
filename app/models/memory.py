from datetime import datetime
from typing import Optional
from beanie import Document, Indexed, PydanticObjectId
from pydantic import Field

class MemoryItem(Document):
    user_id: Indexed(PydanticObjectId)
    document_id: Indexed(PydanticObjectId)
    title: str  # Document title or Concept name
    
    # SM-2 Spaced Repetition stats
    interval: int = 1  # In days
    ease_factor: float = 2.5
    repetitions: int = 0
    next_review: datetime = Field(default_factory=datetime.utcnow)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "memory_cards"
        indexes = [
            "user_id",
            "document_id",
            "next_review"
        ]
