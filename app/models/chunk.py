from beanie import Document, Indexed, PydanticObjectId
from typing import List

class ChunkItem(Document):
    document_id: PydanticObjectId
    user_id: PydanticObjectId
    text: str
    index: int
    embedding: List[float]

    class Settings:
        name = "chunks"
        indexes = [
            "document_id",
            "user_id"
        ]
