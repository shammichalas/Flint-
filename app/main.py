from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.database import init_db
from app.routers import auth, documents, concepts, intelligence

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    await init_db()
    yield
    # Shutdown actions (none needed)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# Configure CORS for Next.js frontend development and production
origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register endpoints under "/api" prefix
app.include_router(auth.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(concepts.router, prefix="/api")
app.include_router(intelligence.router, prefix="/api")

@app.get("/")
def root():
    return {
        "message": "Flint API is running",
        "status": "online"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION
    }
