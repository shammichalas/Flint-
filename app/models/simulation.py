from datetime import datetime
from typing import List, Dict, Any, Optional
from beanie import Document, Indexed, PydanticObjectId
from pydantic import Field

class SimulationItem(Document):
    user_id: Indexed(PydanticObjectId)
    document_id: Indexed(PydanticObjectId)
    hypothesis: str
    predicted_outcome: str
    causal_chain: List[Dict[str, Any]] = []  # List of {"trigger": "...", "impact": "...", "probability": "..."}
    risk_level: str  # "High", "Medium", "Low"
    mitigation_strategies: List[str] = []
    long_term_projection: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "simulations"
        indexes = [
            "user_id",
            "document_id",
            "created_at"
        ]
