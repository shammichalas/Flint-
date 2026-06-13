from datetime import datetime
from typing import List, Optional
from beanie import Document, Indexed, PydanticObjectId
from pydantic import Field

class DocumentItem(Document):
    user_id: PydanticObjectId
    title: str
    filename: str
    file_path: str
    file_size: int
    status: str = "pending"  # "pending", "processing", "completed", "failed"
    error_message: Optional[str] = None
    
    # Ingestion Summaries
    overall_summary: Optional[str] = None
    key_takeaways: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    
    # Phase 4 Compression Layers
    summary_level_4: Optional[str] = None # Detailed Summary (~600-800 words)
    summary_level_2: List[dict] = Field(default_factory=list) # Top 3 Concepts: [{"name": "...", "explanation": "..."}]
    summary_level_1: Optional[str] = None # Core Insight (1-2 sentences)

    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "documents"
        indexes = [
            "user_id",
            "status"
        ]

    def __repr__(self) -> str:
        return f"<Document {self.title} status={self.status}>"
