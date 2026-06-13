from fastapi import APIRouter, Depends, HTTPException, status
from app.core.dependencies import get_current_user
from app.models.user import User
from app.models.concept import ConceptItem
from typing import List

router = APIRouter(prefix="/concepts", tags=["Concepts"])

@router.get("/", response_model=List[dict])
async def get_all_concepts(current_user: User = Depends(get_current_user)):
    concepts = await ConceptItem.find(ConceptItem.user_id == current_user.id).to_list()
    return [
        {
            "id": str(c.id),
            "name": c.name,
            "description": c.description,
            "document_ids": [str(d_id) for d_id in c.document_ids],
            "relations": c.relations
        }
        for c in concepts
    ]

@router.get("/{concept_id}", response_model=dict)
async def get_concept_by_id(concept_id: str, current_user: User = Depends(get_current_user)):
    concept = await ConceptItem.get(concept_id)
    if not concept or concept.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Concept not found in your knowledge base."
        )
    return {
        "id": str(concept.id),
        "name": concept.name,
        "description": concept.description,
        "document_ids": [str(d_id) for d_id in concept.document_ids],
        "relations": concept.relations
    }
