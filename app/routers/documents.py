import os
import asyncio
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, status
from bson import ObjectId
from app.core.dependencies import get_current_user
from app.models.user import User
from app.models.document import DocumentItem
from app.models.chunk import ChunkItem
from app.models.memory import MemoryItem
from app.services.ingestion import start_document_ingestion
from app.core.config import settings
from google import genai
from beanie.operators import In
import math

router = APIRouter(prefix="/documents", tags=["Documents"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload", response_model=DocumentItem, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported."
        )
    
    # Generate unique storage filename
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Save file contents
    file_size = 0
    try:
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                buffer.write(chunk)
                file_size += len(chunk)
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save file to system: {str(e)}"
        )
        
    # Create Document record
    # User.id is a PydanticObjectId (which subclasses ObjectId)
    doc = DocumentItem(
        user_id=current_user.id,
        title=file.filename[:-4] if file.filename.lower().endswith(".pdf") else file.filename,
        filename=file.filename,
        file_path=file_path,
        file_size=file_size,
        status="pending"
    )
    await doc.insert()
    
    # Enqueue background execution task for LLM chunking and summarization
    background_tasks.add_task(start_document_ingestion, str(doc.id))
    
    return doc

@router.get("/", response_model=List[DocumentItem])
async def list_documents(current_user: User = Depends(get_current_user)):
    # Find all documents where user_id matches logged-in user
    docs = await DocumentItem.find(DocumentItem.user_id == current_user.id).sort(-DocumentItem.created_at).to_list()
    return docs

@router.get("/{doc_id}", response_model=DocumentItem)
async def get_document(doc_id: str, current_user: User = Depends(get_current_user)):
    if not ObjectId.is_valid(doc_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format."
        )
        
    doc = await DocumentItem.get(doc_id)
    if not doc or doc.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )
        
    # Self-heal missing Phase 4 compression layers if document is completed but fields are empty
    if doc.status == "completed" and (
        doc.summary_level_4 is None or 
        doc.summary_level_1 is None or 
        not doc.summary_level_2
    ):
        chunks = await ChunkItem.find(ChunkItem.document_id == doc.id).sort(ChunkItem.index).to_list()
        if chunks:
            text = " ".join([c.text for c in chunks])
            try:
                from app.services.ingestion import (
                    generate_level4_detailed_with_gemini,
                    generate_level2_concepts_with_gemini,
                    generate_level1_insight_with_gemini
                )
                
                l4_task = generate_level4_detailed_with_gemini(text) if not doc.summary_level_4 else None
                l2_task = generate_level2_concepts_with_gemini(text) if not doc.summary_level_2 else None
                l1_task = generate_level1_insight_with_gemini(text) if not doc.summary_level_1 else None
                
                tasks = []
                task_indices = {}
                if l4_task:
                    tasks.append(l4_task)
                    task_indices["l4"] = len(tasks) - 1
                if l2_task:
                    tasks.append(l2_task)
                    task_indices["l2"] = len(tasks) - 1
                if l1_task:
                    tasks.append(l1_task)
                    task_indices["l1"] = len(tasks) - 1
                
                if tasks:
                    results = await asyncio.gather(*tasks)
                    if "l4" in task_indices:
                        doc.summary_level_4 = results[task_indices["l4"]]
                    if "l2" in task_indices:
                        doc.summary_level_2 = results[task_indices["l2"]]
                    if "l1" in task_indices:
                        doc.summary_level_1 = results[task_indices["l1"]]
                    
                    await doc.save()
            except Exception as e:
                import logging
                logging.getLogger(__name__).exception("Failed to self-heal compression layers on GET document")
                
    return doc

@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(doc_id: str, current_user: User = Depends(get_current_user)):
    if not ObjectId.is_valid(doc_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format."
        )
        
    doc = await DocumentItem.get(doc_id)
    if not doc or doc.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )
        
    # Delete file from local storage
    if os.path.exists(doc.file_path):
        try:
            os.remove(doc.file_path)
        except Exception:
            pass # Keep going if file deletion fails
            
    await doc.delete()
    return None

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot_product = sum(x * y for x, y in zip(v1, v2))
    magnitude1 = math.sqrt(sum(x * x for x in v1))
    magnitude2 = math.sqrt(sum(x * x for x in v2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

@router.get("/search", response_model=List[dict])
async def search_documents(q: str, current_user: User = Depends(get_current_user)):
    if not q.strip():
        return []
        
    user_docs = await DocumentItem.find(DocumentItem.user_id == current_user.id).to_list()
    if not user_docs:
        return []
        
    doc_ids = [d.id for d in user_docs]
    doc_title_map = {d.id: d.title for d in user_docs}
    
    chunks = await ChunkItem.find(In(ChunkItem.document_id, doc_ids)).to_list()
    if not chunks:
        return []
        
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gemini API Key not configured."
        )
        
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.embed_content(
            model='gemini-embedding-001',
            contents=q
        )
        query_vector = response.embeddings[0].values
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate query embedding: {str(e)}"
        )
        
    results = []
    for chunk in chunks:
        score = cosine_similarity(query_vector, chunk.embedding)
        results.append({
            "chunk_id": str(chunk.id),
            "document_id": str(chunk.document_id),
            "document_title": doc_title_map.get(chunk.document_id, "Unknown Document"),
            "text": chunk.text,
            "index": chunk.index,
            "score": score
        })
        
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:5]

@router.get("/{doc_id}/chunks", response_model=List[dict])
async def get_document_chunks(doc_id: str, current_user: User = Depends(get_current_user)):
    if not ObjectId.is_valid(doc_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format."
        )
        
    doc = await DocumentItem.get(doc_id)
    if not doc or doc.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )
        
    chunks = await ChunkItem.find(ChunkItem.document_id == doc.id).sort(ChunkItem.index).to_list()
    return [
        {
            "id": str(c.id),
            "index": c.index,
            "text": c.text
        }
        for c in chunks
    ]

from app.models.mental_model import MentalModelItem
from pydantic import BaseModel

class GenerateModelRequest(BaseModel):
    model_type: str # "swot", "first_principles", "decision_tree", "cause_effect"

@router.get("/{doc_id}/mental-models", response_model=List[dict])
async def get_document_mental_models(doc_id: str, current_user: User = Depends(get_current_user)):
    if not ObjectId.is_valid(doc_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format."
        )
        
    doc = await DocumentItem.get(doc_id)
    if not doc or doc.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )
        
    models = await MentalModelItem.find(
        MentalModelItem.document_id == doc.id,
        MentalModelItem.user_id == current_user.id
    ).to_list()
    
    return [
        {
            "id": str(m.id),
            "document_id": str(m.document_id),
            "model_type": m.model_type,
            "title": m.title,
            "data": m.data,
            "created_at": m.created_at.isoformat()
        }
        for m in models
    ]

@router.post("/{doc_id}/mental-models", response_model=dict)
async def generate_document_mental_model(
    doc_id: str,
    payload: GenerateModelRequest,
    current_user: User = Depends(get_current_user)
):
    if not ObjectId.is_valid(doc_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format."
        )
        
    doc = await DocumentItem.get(doc_id)
    if not doc or doc.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )
        
    if doc.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document is not fully processed yet."
        )
        
    model_type = payload.model_type.lower()
    if model_type not in ["swot", "first_principles", "decision_tree", "cause_effect"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid mental model type. Choose from: swot, first_principles, decision_tree, cause_effect"
        )
        
    existing = await MentalModelItem.find_one(
        MentalModelItem.document_id == doc.id,
        MentalModelItem.user_id == current_user.id,
        MentalModelItem.model_type == model_type
    )
    if existing:
        return {
            "id": str(existing.id),
            "document_id": str(existing.document_id),
            "model_type": existing.model_type,
            "title": existing.title,
            "data": existing.data,
            "created_at": existing.created_at.isoformat()
        }
        
    chunks = await ChunkItem.find(ChunkItem.document_id == doc.id).sort(ChunkItem.index).to_list()
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No source chunks found to reconstruct document text."
        )
        
    text = " ".join([c.text for c in chunks])
    
    try:
        from app.services.ingestion import (
            generate_swot_with_gemini,
            generate_first_principles_with_gemini,
            generate_decision_tree_with_gemini,
            generate_cause_effect_with_gemini
        )
        
        title = ""
        model_data = {}
        
        if model_type == "swot":
            title = f"SWOT Deconstruction: {doc.title}"
            model_data = await generate_swot_with_gemini(text)
        elif model_type == "first_principles":
            title = f"First Principles Mapping: {doc.title}"
            model_data = await generate_first_principles_with_gemini(text)
        elif model_type == "decision_tree":
            title = f"Decision Logic Flow: {doc.title}"
            model_data = await generate_decision_tree_with_gemini(text)
        elif model_type == "cause_effect":
            title = f"Causal Loop Mapping: {doc.title}"
            model_data = await generate_cause_effect_with_gemini(text)
            
        new_model = MentalModelItem(
            document_id=doc.id,
            user_id=current_user.id,
            model_type=model_type,
            title=title,
            data=model_data
        )
        await new_model.insert()
        
        return {
            "id": str(new_model.id),
            "document_id": str(new_model.document_id),
            "model_type": new_model.model_type,
            "title": new_model.title,
            "data": new_model.data,
            "created_at": new_model.created_at.isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gemini mental model extraction failed: {str(e)}"
        )
from datetime import datetime, timedelta

@router.get("/memory/stats", response_model=dict)
async def get_memory_stats(current_user: User = Depends(get_current_user)):
    now = datetime.utcnow()
    total_cards = await MemoryItem.find(MemoryItem.user_id == current_user.id).count()
    due_cards = await MemoryItem.find(
        MemoryItem.user_id == current_user.id,
        MemoryItem.next_review <= now
    ).count()
    
    cards = await MemoryItem.find(MemoryItem.user_id == current_user.id).to_list()
    avg_ease = sum(c.ease_factor for c in cards) / len(cards) if cards else 2.5
    avg_retention = min(100, max(50, round((avg_ease / 2.5) * 90))) if cards else 100

    study_deck = [
        {
            "id": str(c.id),
            "document_id": str(c.document_id),
            "title": c.title,
            "interval": c.interval,
            "next_review": c.next_review.isoformat(),
            "is_due": c.next_review <= now
        }
        for c in cards
    ]

    return {
        "total_cards": total_cards,
        "due_cards": due_cards,
        "avg_retention": avg_retention,
        "study_deck": study_deck
    }

@router.get("/{doc_id}/quiz", response_model=dict)
async def get_document_quiz(doc_id: str, current_user: User = Depends(get_current_user)):
    if not ObjectId.is_valid(doc_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format."
        )
        
    doc = await DocumentItem.get(doc_id)
    if not doc or doc.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )
        
    if doc.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document is not fully processed yet."
        )
        
    chunks = await ChunkItem.find(ChunkItem.document_id == doc.id).sort(ChunkItem.index).to_list()
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No source chunks found to reconstruct document text."
        )
        
    text = " ".join([c.text for c in chunks])
    
    try:
        from app.services.ingestion import generate_document_quiz_with_gemini
        quiz_data = await generate_document_quiz_with_gemini(text)
        return quiz_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate active recall quiz: {str(e)}"
        )

class QuizSubmissionRequest(BaseModel):
    score: int  # 0 to 3

@router.post("/{doc_id}/quiz/submit", response_model=dict)
async def submit_document_quiz(
    doc_id: str,
    payload: QuizSubmissionRequest,
    current_user: User = Depends(get_current_user)
):
    if not ObjectId.is_valid(doc_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format."
        )
        
    doc = await DocumentItem.get(doc_id)
    if not doc or doc.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )
        
    score = payload.score
    if not (0 <= score <= 3):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Score must be between 0 and 3."
        )
        
    if score == 3:
        q = 5
    elif score == 2:
        q = 4
    elif score == 1:
        q = 2
    else:
        q = 0

    card = await MemoryItem.find_one(
        MemoryItem.document_id == doc.id,
        MemoryItem.user_id == current_user.id
    )
    if not card:
        card = MemoryItem(
            document_id=doc.id,
            user_id=current_user.id,
            title=doc.title,
            interval=1,
            ease_factor=2.5,
            repetitions=0
        )
        
    if q < 3:
        card.repetitions = 0
        card.interval = 1
    else:
        if card.repetitions == 0:
            card.interval = 1
        elif card.repetitions == 1:
            card.interval = 6
        else:
            card.interval = max(1, round(card.interval * card.ease_factor))
        card.repetitions += 1
        
    card.ease_factor = card.ease_factor + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    if card.ease_factor < 1.3:
        card.ease_factor = 1.3
        
    card.next_review = datetime.utcnow() + timedelta(days=card.interval)
    await card.save()
    
    return {
        "document_id": str(card.document_id),
        "title": card.title,
        "interval": card.interval,
        "ease_factor": card.ease_factor,
        "repetitions": card.repetitions,
        "next_review": card.next_review.isoformat(),
        "score": score
    }
