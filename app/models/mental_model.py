from datetime import datetime
from typing import Dict, Any
from beanie import Document, Indexed, PydanticObjectId
from pydantic import Field

class MentalModelItem(Document):
    document_id: Indexed(PydanticObjectId)
    user_id: Indexed(PydanticObjectId)
    model_type: str  # "swot", "first_principles", "decision_tree", "cause_effect"
    title: str
    data: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "mental_models"
        indexes = [
            "document_id",
            "user_id",
            "model_type"
        ]
