from beanie import Document, Indexed, PydanticObjectId
from typing import List, Dict, Any

class ConceptItem(Document):
    user_id: PydanticObjectId
    name: str
    description: str
    document_ids: List[PydanticObjectId]
    relations: List[Dict[str, Any]] = []

    class Settings:
        name = "concepts"
        indexes = [
            "user_id",
            "name"
        ]
