from motor.motor_asyncio import AsyncIOMotorClient
import motor.core
from beanie import init_beanie
from app.core.config import settings
from app.models.user import User
from app.models.document import DocumentItem
from app.models.chunk import ChunkItem
from app.models.concept import ConceptItem
from app.models.mental_model import MentalModelItem
from app.models.memory import MemoryItem
from app.models.simulation import SimulationItem



# Monkey-patch Motor to prevent Beanie v2's check for callable append_metadata on client database objects
try:
    if hasattr(motor.core.AgnosticDatabase, "__call__"):
        delattr(motor.core.AgnosticDatabase, "__call__")
except Exception:
    pass

async def init_db():
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    database = client[settings.DATABASE_NAME]
    
    # Initialize Beanie with document models
    await init_beanie(
        database=database,
        document_models=[
            User,
            DocumentItem,
            ChunkItem,
            ConceptItem,
            MentalModelItem,
            MemoryItem,
            SimulationItem
        ]


    )
