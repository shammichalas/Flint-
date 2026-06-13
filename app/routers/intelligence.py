import os
import math
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from bson import ObjectId
from pydantic import BaseModel, Field

from app.core.dependencies import get_current_user
from app.models.user import User
from app.models.document import DocumentItem
from app.models.chunk import ChunkItem
from app.models.concept import ConceptItem
from app.models.memory import MemoryItem
from app.models.simulation import SimulationItem
from app.core.config import settings

from google import genai
from beanie.operators import In

from app.services.ingestion import (
    generate_simulation_with_gemini,
    generate_tutor_response,
    generate_cross_document_synthesis
)

router = APIRouter(prefix="/intelligence", tags=["Intelligence"])

# Cosine similarity helper
def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot_product = sum(x * y for x, y in zip(v1, v2))
    magnitude1 = math.sqrt(sum(x * x for x in v1))
    magnitude2 = math.sqrt(sum(x * x for x in v2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

class SimulationRequest(BaseModel):
    hypothesis: str

class CrossReasoningRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None

class TutorChatRequest(BaseModel):
    message: str
    chat_history: List[dict] = []
    doc_id: Optional[str] = None


@router.post("/documents/{doc_id}/simulate", response_model=dict)
async def run_simulation(
    doc_id: str,
    payload: SimulationRequest,
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
            detail="Document must be fully processed before running simulations."
        )
        
    # Get all chunks
    chunks = await ChunkItem.find(ChunkItem.document_id == doc.id).sort(ChunkItem.index).to_list()
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No source chunks found for this document."
        )
        
    text = " ".join([c.text for c in chunks])
    
    try:
        sim_data = await generate_simulation_with_gemini(text, payload.hypothesis)
        
        sim = SimulationItem(
            user_id=current_user.id,
            document_id=doc.id,
            hypothesis=payload.hypothesis,
            predicted_outcome=sim_data.get("predicted_outcome", ""),
            causal_chain=sim_data.get("causal_chain", []),
            risk_level=sim_data.get("risk_level", "Medium"),
            mitigation_strategies=sim_data.get("mitigation_strategies", []),
            long_term_projection=sim_data.get("long_term_projection", "")
        )
        await sim.insert()
        
        return {
            "id": str(sim.id),
            "document_id": str(sim.document_id),
            "hypothesis": sim.hypothesis,
            "predicted_outcome": sim.predicted_outcome,
            "causal_chain": sim.causal_chain,
            "risk_level": sim.risk_level,
            "mitigation_strategies": sim.mitigation_strategies,
            "long_term_projection": sim.long_term_projection,
            "created_at": sim.created_at.isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation modeling failed: {str(e)}"
        )


@router.get("/documents/{doc_id}/simulations", response_model=List[dict])
async def list_simulations(
    doc_id: str,
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
        
    sims = await SimulationItem.find(
        SimulationItem.document_id == doc.id,
        SimulationItem.user_id == current_user.id
    ).sort(-SimulationItem.created_at).to_list()
    
    return [
        {
            "id": str(s.id),
            "document_id": str(s.document_id),
            "hypothesis": s.hypothesis,
            "predicted_outcome": s.predicted_outcome,
            "causal_chain": s.causal_chain,
            "risk_level": s.risk_level,
            "mitigation_strategies": s.mitigation_strategies,
            "long_term_projection": s.long_term_projection,
            "created_at": s.created_at.isoformat()
        }
        for s in sims
    ]


@router.post("/cross-reasoning", response_model=dict)
async def cross_document_reasoning(
    payload: CrossReasoningRequest,
    current_user: User = Depends(get_current_user)
):
    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query query cannot be empty."
        )
        
    # Find active documents of current user
    user_docs = await DocumentItem.find(
        DocumentItem.user_id == current_user.id,
        DocumentItem.status == "completed"
    ).to_list()
    
    if not user_docs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You do not have any completed documents to query."
        )
        
    # Filter by requested documents if provided
    if payload.document_ids:
        filtered_ids = [ObjectId(d_id) for d_id in payload.document_ids if ObjectId.is_valid(d_id)]
        user_docs = [d for d in user_docs if d.id in filtered_ids]
        if not user_docs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="None of the specified document IDs were found."
            )
            
    doc_ids = [d.id for d in user_docs]
    doc_title_map = {d.id: d.title for d in user_docs}
    
    # Generate query embedding
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
            contents=payload.query
        )
        query_vector = response.embeddings[0].values
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}"
        )
        
    # Retrieve all chunks
    chunks = await ChunkItem.find(In(ChunkItem.document_id, doc_ids)).to_list()
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No source fragments found to compare."
        )
        
    # Match using cosine similarity
    scored_chunks = []
    for chunk in chunks:
        score = cosine_similarity(query_vector, chunk.embedding)
        scored_chunks.append({
            "document_id": str(chunk.document_id),
            "document_title": doc_title_map.get(chunk.document_id, "Unknown"),
            "text": chunk.text,
            "score": score
        })
        
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = scored_chunks[:10]  # Take top 10 chunks for rich cross-doc context
    
    try:
        synthesis_data = await generate_cross_document_synthesis(top_chunks, payload.query)
        
        # Format citations
        citations = []
        seen_citations = set()
        for chunk in top_chunks:
            citation_key = (chunk["document_title"], chunk["text"][:100])
            if chunk["score"] > 0.4 and citation_key not in seen_citations:
                citations.append({
                    "document_id": chunk["document_id"],
                    "document_title": chunk["document_title"],
                    "snippet": chunk["text"],
                    "score": chunk["score"]
                })
                seen_citations.add(citation_key)
                
        return {
            "synthesis": synthesis_data.get("synthesis", ""),
            "comparisons": synthesis_data.get("comparisons", []),
            "citations": citations[:4]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cross-document reasoning failed: {str(e)}"
        )


@router.post("/tutor/chat", response_model=dict)
async def tutor_chat(
    payload: TutorChatRequest,
    current_user: User = Depends(get_current_user)
):
    if not payload.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty."
        )
        
    # Get active documents
    user_docs = await DocumentItem.find(
        DocumentItem.user_id == current_user.id,
        DocumentItem.status == "completed"
    ).to_list()
    
    if not user_docs:
        # Fallback to general AI answer if no documents exist
        doc_ids = []
    else:
        if payload.doc_id and ObjectId.is_valid(payload.doc_id):
            doc_ids = [ObjectId(payload.doc_id)]
        else:
            doc_ids = [d.id for d in user_docs]
            
    # If there are documents, fetch relevant chunks
    context_texts = []
    if doc_ids:
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
                contents=payload.message
            )
            query_vector = response.embeddings[0].values
            
            chunks = await ChunkItem.find(In(ChunkItem.document_id, doc_ids)).to_list()
            
            scored_chunks = []
            for chunk in chunks:
                score = cosine_similarity(query_vector, chunk.embedding)
                scored_chunks.append((score, chunk.text))
                
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            # Take top 5
            context_texts = [item[1] for item in scored_chunks[:5]]
        except Exception:
            pass # Keep going with empty context if vector search fails
            
    try:
        reply = await generate_tutor_response(context_texts, payload.message, payload.chat_history)
        return {"response": reply}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tutoring session failed: {str(e)}"
        )


@router.get("/dashboard/stats", response_model=dict)
async def get_dashboard_stats(current_user: User = Depends(get_current_user)):
    now = datetime.utcnow()
    
    # 1. Base counts
    total_docs = await DocumentItem.find(
        DocumentItem.user_id == current_user.id,
        DocumentItem.status == "completed"
    ).count()
    
    total_concepts = await ConceptItem.find(
        ConceptItem.user_id == current_user.id
    ).count()
    
    total_cards = await MemoryItem.find(
        MemoryItem.user_id == current_user.id
    ).count()
    
    due_cards = await MemoryItem.find(
        MemoryItem.user_id == current_user.id,
        MemoryItem.next_review <= now
    ).count()
    
    # Calculate retention score
    cards = await MemoryItem.find(MemoryItem.user_id == current_user.id).to_list()
    avg_ease = sum(c.ease_factor for c in cards) / len(cards) if cards else 2.5
    avg_retention = min(100, max(50, round((avg_ease / 2.5) * 90))) if cards else 100
    
    # 2. Topic Categorization (dynamic mapping from document titles)
    categories = {
        "Technology & Software": 0,
        "Business & Strategy": 0,
        "Science & Research": 0,
        "Finance & Economics": 0,
        "Healthcare & Biotech": 0,
        "General Knowledge": 0
    }
    
    docs = await DocumentItem.find(
        DocumentItem.user_id == current_user.id,
        DocumentItem.status == "completed"
    ).to_list()
    
    for d in docs:
        title_lower = d.title.lower()
        if any(w in title_lower for w in ["code", "software", "api", "tech", "web", "program", "app", "model", "network"]):
            categories["Technology & Software"] += 1
        elif any(w in title_lower for w in ["business", "strategy", "market", "management", "plan", "product", "sales"]):
            categories["Business & Strategy"] += 1
        elif any(w in title_lower for w in ["science", "physics", "chemistry", "research", "paper", "theory"]):
            categories["Science & Research"] += 1
        elif any(w in title_lower for w in ["finance", "economy", "money", "stock", "pricing", "budget", "tax"]):
            categories["Finance & Economics"] += 1
        elif any(w in title_lower for w in ["health", "medical", "bio", "clinical", "patient", "drug"]):
            categories["Healthcare & Biotech"] += 1
        else:
            categories["General Knowledge"] += 1
            
    # Keep only categories with > 0 documents, or general if all 0
    categories = {k: v for k, v in categories.items() if v > 0}
    if not categories:
        categories = {"General Knowledge": 0}
        
    # 3. Timeline milestones
    timeline = []
    # Get last 3 documents
    recent_docs = await DocumentItem.find(
        DocumentItem.user_id == current_user.id
    ).sort(-DocumentItem.created_at).limit(3).to_list()
    for d in recent_docs:
        timeline.append({
            "type": "document",
            "title": f"Archived '{d.title}'",
            "timestamp": d.created_at.isoformat(),
            "details": f"Ingested {d.filename} ({round(d.file_size/1024, 1)} KB)"
        })
        
    # Get last 3 simulations
    recent_sims = await SimulationItem.find(
        SimulationItem.user_id == current_user.id
    ).sort(-SimulationItem.created_at).limit(3).to_list()
    for s in recent_sims:
        doc = next((d for d in docs if d.id == s.document_id), None)
        doc_title = doc.title if doc else "Document"
        timeline.append({
            "type": "simulation",
            "title": f"Simulated on '{doc_title}'",
            "timestamp": s.created_at.isoformat(),
            "details": f"Hypothesis: \"{s.hypothesis[:60]}...\""
        })
        
    # Get last 3 recall card creations
    recent_cards = await MemoryItem.find(
        MemoryItem.user_id == current_user.id
    ).sort(-MemoryItem.created_at).limit(3).to_list()
    for c in recent_cards:
        timeline.append({
            "type": "memory",
            "title": f"Registered review card",
            "timestamp": c.created_at.isoformat(),
            "details": f"Added memory tracking for '{c.title}'"
        })
        
    # Sort timeline by timestamp desc and limit to 5
    timeline.sort(key=lambda x: x["timestamp"], reverse=True)
    timeline = timeline[:5]
    
    return {
        "total_documents": total_docs,
        "total_concepts": total_concepts,
        "total_memory_cards": total_cards,
        "due_memory_cards": due_cards,
        "avg_retention": avg_retention,
        "categories": categories,
        "timeline": timeline
    }
