import os
import logging
from pypdf import PdfReader
from app.models.document import DocumentItem
from app.core.config import settings
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
import json
from datetime import datetime

from app.models.chunk import ChunkItem
from app.models.concept import ConceptItem

logger = logging.getLogger(__name__)

class SummaryResponse(BaseModel):
    overall_summary: str = Field(description="A concise 1-2 paragraph summary of the document.")
    key_takeaways: List[str] = Field(description="A list of the top 5-10 key takeaways or conceptual pillars.")
    action_items: List[str] = Field(description="A list of practical action items, decisions, or next steps suggested by the text.")

class ConceptRelation(BaseModel):
    target: str = Field(description="The name of the target concept it connects to.")
    relation_type: str = Field(description="The type of connection, e.g., 'defines', 'utilizes', 'prerequisite_of', 'related_to'.")
    description: str = Field(description="Brief 1-sentence explanation of the relationship.")

class ConceptExtracted(BaseModel):
    name: str = Field(description="The name of the concept. Use proper title case.")
    description: str = Field(description="A concise 1-2 sentence description defining the concept.")
    relations: List[ConceptRelation] = Field(default_factory=list, description="List of connections from this concept to other concepts in the text.")

class ConceptExtractionResponse(BaseModel):
    concepts: List[ConceptExtracted] = Field(description="A list of core concepts extracted from the text.")

class ConceptPillar(BaseModel):
    name: str = Field(description="Name of the conceptual pillar. Use proper title case.")
    explanation: str = Field(description="A concise 2-sentence explanation of what it is and why it's critical.")

class Level2ConceptsResponse(BaseModel):
    concepts: List[ConceptPillar] = Field(description="List of exactly 3 core conceptual pillars or themes.")

class SwotResponse(BaseModel):
    strengths: List[str] = Field(description="3-5 key strengths or internal positive factors.")
    weaknesses: List[str] = Field(description="3-5 key weaknesses or internal negative factors.")
    opportunities: List[str] = Field(description="3-5 external opportunities or positive possibilities.")
    threats: List[str] = Field(description="3-5 external threats, risks, or challenges.")

class AssumptionPair(BaseModel):
    assumption: str = Field(description="The status-quo assumption commonly believed.")
    reality: str = Field(description="The underlying objective reality that challenges this assumption.")
    
class FirstPrinciplesResponse(BaseModel):
    core_problem: str = Field(description="The main problem being solved, stated in a simple form.")
    assumptions_challenged: List[AssumptionPair] = Field(description="2-3 standard assumptions challenged by the text.")
    fundamental_truths: List[str] = Field(description="3-4 raw, indisputable facts or truths extracted from the text.")
    new_solution: str = Field(description="The new solution constructed directly from these fundamental truths.")

class DecisionNode(BaseModel):
    id: str = Field(description="Unique short string ID for the decision node.")
    label: str = Field(description="The condition, choice, or event (e.g. 'If user clicks X' or 'Under high demand').")
    result: str = Field(description="The outcome, recommendation, or next action step.")
    
class DecisionTreeResponse(BaseModel):
    root_question: str = Field(description="The core starting decision or question (e.g., 'How to optimize X?').")
    nodes: List[DecisionNode] = Field(description="List of logical decision branches and their results.")

class SystemConnection(BaseModel):
    source: str = Field(description="The starting variable or factor (e.g., 'User engagement').")
    target: str = Field(description="The impacted variable or factor (e.g., 'Retention rate').")
    direction: str = Field(description="Either 'positive' (increase leads to increase) or 'negative' (increase leads to decrease).")
    description: str = Field(description="1-sentence explanation of how the cause leads to the effect.")
    
class CauseEffectResponse(BaseModel):
    factors: List[str] = Field(description="List of key variables or system factors (e.g., ['Prices', 'Sales', 'Profit']).")
    connections: List[SystemConnection] = Field(description="List of directed links showing feedback loops or causal relations.")

class QuizQuestion(BaseModel):
    question: str = Field(description="The active recall question based on core facts/methods.")
    options: List[str] = Field(description="Exactly 4 multiple-choice options.")
    correct_answer: str = Field(description="The exact text of the correct option matching one of the options.")
    explanation: str = Field(description="1-2 sentences explaining why this answer is correct based on the text.")

class QuizResponse(BaseModel):
    questions: List[QuizQuestion] = Field(description="A list of exactly 3 multiple-choice active recall questions.")




def extract_text_from_pdf(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at: {file_path}")
    
    reader = PdfReader(file_path)
    text = ""
    for idx, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def chunk_text(text: str, target_size: int = 1000, overlap: int = 200) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = []
    current_size = 0
    
    for p in paragraphs:
        current_chunk.append(p)
        current_size += len(p) + 1
        
        if current_size >= target_size:
            chunks.append("\n".join(current_chunk))
            last_p = current_chunk[-1]
            if len(last_p) < overlap:
                current_chunk = [last_p]
                current_size = len(last_p)
            else:
                current_chunk = []
                current_size = 0
                
    if current_chunk:
        chunks.append("\n".join(current_chunk))
        
    return chunks

import asyncio

async def generate_summary_with_gemini(text: str) -> dict:
    """
    Calls Google Gemini API (gemini-2.5-flash) to produce structured JSON summaries.
    """
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set. Ingestion cannot proceed.")
    
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    Analyze the following extracted document text and produce a structured analysis.
    
    DOCUMENT TEXT:
    {text[:45000]}
    """
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SummaryResponse,
                    system_instruction="You are an expert thought compression agent. Your job is to extract raw document contents and distill them into core insights, summaries, and action steps."
                ),
            )
            result_data = json.loads(response.text)
            return result_data
        except Exception as e:
            if attempt == 2:
                raise e
            await asyncio.sleep(2 * (attempt + 1))

async def extract_concepts_with_gemini(text: str) -> dict:
    """
    Extracts key concepts and relationship maps from the document using gemini-2.5-flash.
    """
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    Analyze the following document text and extract the top 5-10 core concepts, key terms, or intellectual pillars discussed.
    For each concept, provide a clear definition/description and identify any relationships it has with other concepts within the text.
    
    DOCUMENT TEXT:
    {text[:45000]}
    """
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ConceptExtractionResponse,
                    system_instruction="You are an expert knowledge graph engineer. Your task is to extract concepts and their semantic relationships from text to build a robust mental model."
                ),
            )
            result_data = json.loads(response.text)
            return result_data
        except Exception as e:
            if attempt == 2:
                raise e
            await asyncio.sleep(2 * (attempt + 1))

async def generate_level4_detailed_with_gemini(text: str) -> str:
    """
    Calls Google Gemini API (gemini-2.5-flash) to produce a comprehensive, detailed markdown summary (~600-800 words).
    """
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    
    client = genai.Client(api_key=api_key)
    prompt = f"""
    Analyze the following document text and write a highly detailed, comprehensive summary (approximately 600-800 words).
    Cover all primary sections, key arguments, data, methodologies, and conclusions. Use rich markdown formatting with headings, bullet points, and bold text.
    
    DOCUMENT TEXT:
    {text[:45000]}
    """
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are an expert technical writer. Write deep, structured summaries that retain maximum factual information."
                ),
            )
            return response.text.strip()
        except Exception as e:
            if attempt == 2:
                raise e
            await asyncio.sleep(2 * (attempt + 1))

async def generate_level2_concepts_with_gemini(text: str) -> List[dict]:
    """
    Calls Google Gemini API (gemini-2.5-flash) to extract exactly the top 3 core conceptual pillars of the document.
    """
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    
    client = genai.Client(api_key=api_key)
    prompt = f"""
    Analyze the following document text and extract exactly the top 3 most fundamental conceptual pillars, core themes, or intellectual foundations.
    For each pillar, provide its name and a clear 2-sentence explanation of what it is and why it matters in this context.
    
    DOCUMENT TEXT:
    {text[:45000]}
    """
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=Level2ConceptsResponse,
                    system_instruction="You are a senior research analyst. Focus only on the absolute most important conceptual pillars of the text."
                ),
            )
            result_data = json.loads(response.text)
            return result_data.get("concepts", [])
        except Exception as e:
            if attempt == 2:
                raise e
            await asyncio.sleep(2 * (attempt + 1))

async def generate_level1_insight_with_gemini(text: str) -> str:
    """
    Calls Google Gemini API (gemini-2.5-flash) to distill the text into a single, profound, action-oriented core insight (1-2 sentences).
    """
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    
    client = genai.Client(api_key=api_key)
    prompt = f"""
    Analyze the following document text and distill its entire message into a single, profound, action-oriented core insight (1-2 sentences).
    What is the absolute core takeaway or thesis that summarizing everything else?
    
    DOCUMENT TEXT:
    {text[:45000]}
    """
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are a philosophical insight distiller. Find the one ultimate takeaway of any text and express it in a powerful, dense statement."
                ),
            )
            return response.text.strip()
        except Exception as e:
            if attempt == 2:
                raise e
            await asyncio.sleep(2 * (attempt + 1))

async def generate_swot_with_gemini(text: str) -> dict:
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    
    client = genai.Client(api_key=api_key)
    prompt = f"""
    Analyze the following document text and perform a comprehensive SWOT analysis (Strengths, Weaknesses, Opportunities, Threats) from the context of the document.
    
    DOCUMENT TEXT:
    {text[:45000]}
    """
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SwotResponse,
                    system_instruction="You are a senior business strategy consultant. Generate structured SWOT data."
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            if attempt == 2:
                raise e
            await asyncio.sleep(2 * (attempt + 1))

async def generate_first_principles_with_gemini(text: str) -> dict:
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    
    client = genai.Client(api_key=api_key)
    prompt = f"""
    Analyze the following document text and perform a First Principles deconstruction:
    1. Identify the core problem.
    2. List standard status-quo assumptions challenged by this text.
    3. List 3-4 fundamental objective facts or truths established.
    4. Provide the reconstructed solution built from these truths.
    
    DOCUMENT TEXT:
    {text[:45000]}
    """
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=FirstPrinciplesResponse,
                    system_instruction="You are a philosophical problem solver. Break down concepts into their absolute fundamental facts and build new solutions from there."
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            if attempt == 2:
                raise e
            await asyncio.sleep(2 * (attempt + 1))

async def generate_decision_tree_with_gemini(text: str) -> dict:
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    
    client = genai.Client(api_key=api_key)
    prompt = f"""
    Analyze the following document text and extract the core logical decisions, decision-making branches, or standard options discussed.
    Formulate them as a structured decision tree with a starting root question and conditional branches (nodes) showing options and outcomes.
    
    DOCUMENT TEXT:
    {text[:45000]}
    """
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DecisionTreeResponse,
                    system_instruction="You are a logic engineer. Structure decision branches cleanly as conditional choices and outcomes."
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            if attempt == 2:
                raise e
            await asyncio.sleep(2 * (attempt + 1))

async def generate_cause_effect_with_gemini(text: str) -> dict:
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    
    client = genai.Client(api_key=api_key)
    prompt = f"""
    Analyze the following document text and map out the dynamic cause-and-effect connections or systems feedback loops.
    Extract the key variables/factors and list the directed connections showing how variables increase or decrease one another.
    
    DOCUMENT TEXT:
    {text[:45000]}
    """
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CauseEffectResponse,
                    system_instruction="You are a systems dynamics analyst. Focus on variables, feedback loops, and directed causal relationships."
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            if attempt == 2:
                raise e
            await asyncio.sleep(2 * (attempt + 1))

async def generate_document_quiz_with_gemini(text: str) -> dict:
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    
    client = genai.Client(api_key=api_key)
    prompt = f"""
    Analyze the following document text and generate exactly 3 multiple-choice active recall questions based on its core themes, facts, and conclusions.
    For each question, provide exactly 4 distinct options, specify which option text is the correct answer, and write a brief explanation.
    
    DOCUMENT TEXT:
    {text[:45000]}
    """
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=QuizResponse,
                    system_instruction="You are an expert educator. Create high-quality active recall questions to verify student comprehension."
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            if attempt == 2:
                raise e
            await asyncio.sleep(2 * (attempt + 1))

async def start_document_ingestion(doc_id: str):



    # Retrieve the document from database
    doc = await DocumentItem.get(doc_id)
    if not doc:
        logger.error(f"Ingestion failed: Document {doc_id} not found in database.")
        return
        
    # Update status to processing
    doc.status = "processing"
    doc.updated_at = datetime.utcnow()
    await doc.save()
    
    try:
        # Step 1: Extract text
        text = extract_text_from_pdf(doc.file_path)
        if not text.strip():
            raise ValueError("Document appears to be empty or contains no extractable text.")
            
        # Step 2: Call Gemini API for summaries & compression layers in parallel
        summary_task = generate_summary_with_gemini(text)
        level4_task = generate_level4_detailed_with_gemini(text)
        level2_task = generate_level2_concepts_with_gemini(text)
        level1_task = generate_level1_insight_with_gemini(text)
        
        summary_result, l4_summary, l2_concepts, l1_insight = await asyncio.gather(
            summary_task, level4_task, level2_task, level1_task
        )
        
        doc.overall_summary = summary_result.get("overall_summary", "No summary generated.")
        doc.key_takeaways = summary_result.get("key_takeaways", [])
        doc.action_items = summary_result.get("action_items", [])
        
        doc.summary_level_4 = l4_summary
        doc.summary_level_2 = l2_concepts
        doc.summary_level_1 = l1_insight

        
        # Step 3: Chunking & Embeddings
        chunks = chunk_text(text)
        batch_size = 30
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key)
            
            response = client.models.embed_content(
                model='gemini-embedding-001',
                contents=batch_chunks
            )
            
            for j, val in enumerate(response.embeddings):
                chunk_idx = i + j
                chunk_text_val = batch_chunks[j]
                embedding_vector = val.values
                
                chunk_item = ChunkItem(
                    document_id=doc.id,
                    user_id=doc.user_id,
                    text=chunk_text_val,
                    index=chunk_idx,
                    embedding=embedding_vector
                )
                await chunk_item.insert()
                
        # Step 4: Concept & Relationship Extraction
        concepts_data = await extract_concepts_with_gemini(text)
        for extracted_concept in concepts_data.get("concepts", []):
            name = extracted_concept.get("name", "").strip()
            if not name:
                continue
                
            existing_concept = await ConceptItem.find_one(
                ConceptItem.user_id == doc.user_id,
                ConceptItem.name == name
            )
            
            new_relations = extracted_concept.get("relations", [])
            # Format relations as dicts matching the DB schema
            formatted_relations = []
            for r in new_relations:
                formatted_relations.append({
                    "target": r.get("target", ""),
                    "type": r.get("relation_type", ""),
                    "description": r.get("description", "")
                })
            
            if existing_concept:
                if doc.id not in existing_concept.document_ids:
                    existing_concept.document_ids.append(doc.id)
                
                # Merge relationships
                for fr in formatted_relations:
                    target_name = fr.get("target", "").strip()
                    if not target_name:
                        continue
                    exists = False
                    for existing_rel in existing_concept.relations:
                        if existing_rel.get("target", "").lower() == target_name.lower():
                            exists = True
                            break
                    if not exists:
                        existing_concept.relations.append(fr)
                        
                await existing_concept.save()
            else:
                new_concept = ConceptItem(
                    user_id=doc.user_id,
                    name=name,
                    description=extracted_concept.get("description", ""),
                    document_ids=[doc.id],
                    relations=formatted_relations
                )
                await new_concept.insert()

        doc.status = "completed"
        doc.updated_at = datetime.utcnow()
        await doc.save()
        logger.info(f"Successfully ingested and summarized document {doc_id}")
        
    except Exception as e:
        logger.exception(f"Error ingesting document {doc_id}")
        doc.status = "failed"
        doc.error_message = str(e)
        doc.updated_at = datetime.utcnow()
        await doc.save()


class CausalFactor(BaseModel):
    trigger: str = Field(description="The event or condition.")
    impact: str = Field(description="The effect or outcome.")
    probability: str = Field(description="Likelihood of occurrence: 'High', 'Medium', or 'Low'.")

class SimulationResponse(BaseModel):
    predicted_outcome: str = Field(description="Detailed prediction of the overall outcome.")
    causal_chain: List[CausalFactor] = Field(description="Step-by-step causal chain showing dynamic flow.")
    risk_level: str = Field(description="Overall risk level: 'High', 'Medium', or 'Low'.")
    mitigation_strategies: List[str] = Field(description="3-4 practical mitigation strategies.")
    long_term_projection: str = Field(description="Description of long-term behaviors or trends.")

async def generate_simulation_with_gemini(text: str, hypothesis: str) -> dict:
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    
    client = genai.Client(api_key=api_key)
    prompt = f"""
    You are modeling a scenario in the context of the following document.
    Analyze the document text and evaluate this specific hypothesis or scenario:
    
    HYPOTHESIS/SCENARIO:
    {hypothesis}
    
    DOCUMENT TEXT:
    {text[:45000]}
    """
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SimulationResponse,
                    system_instruction="You are a senior simulation lab expert and systems analyst. Predict the dynamic outcome and causal impacts of the hypothesis based on document facts."
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            if attempt == 2:
                raise e
            await asyncio.sleep(2 * (attempt + 1))


async def generate_tutor_response(context_chunks: List[str], message: str, chat_history: List[dict]) -> str:
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
        
    client = genai.Client(api_key=api_key)
    
    formatted_context = "\n---\n".join(context_chunks)
    
    history_str = ""
    for turn in chat_history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        role_label = "Student" if role == "user" else "Tutor"
        history_str += f"{role_label}: {content}\n"
        
    prompt = f"""
    You are an AI Tutor helping a student understand a set of documents.
    Here is the reference context from the documents:
    {formatted_context[:30000]}
    
    CHAT HISTORY SO FAR:
    {history_str}
    
    STUDENT MESSAGE:
    {message}
    
    Provide your response as a supportive, encouraging, and highly intelligent AI tutor.
    Use vivid analogies to explain complex terms, highlight key concepts in bold, and offer a short follow-up question at the end to check their understanding. Keep it structured in markdown.
    """
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are a brilliant personal AI tutor. Break down complex concepts into digestible insights with analogies and interactive checks."
                ),
            )
            return response.text.strip()
        except Exception as e:
            if attempt == 2:
                raise e
            await asyncio.sleep(2 * (attempt + 1))


class ComparativeAspect(BaseModel):
    aspect: str = Field(description="The dimension or concept being compared.")
    findings: str = Field(description="Summary of how the different documents compare on this aspect.")

class CrossDocumentResponse(BaseModel):
    synthesis: str = Field(description="Markdown formatted comparative synthesis explaining similarities, differences, and key takeaways across the source texts.")
    comparisons: List[ComparativeAspect] = Field(description="Aspect-by-aspect breakdown of comparison.")

async def generate_cross_document_synthesis(context_chunks: List[dict], query: str) -> dict:
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    
    client = genai.Client(api_key=api_key)
    
    formatted_context = ""
    for idx, chunk in enumerate(context_chunks):
        formatted_context += f"\n[Document #{idx+1}]: {chunk.get('document_title', 'Unknown')}\nContent: {chunk.get('text', '')}\n---\n"
        
    prompt = f"""
    You are performing cross-document reasoning.
    Here is the query/question:
    {query}
    
    Here are the source fragments from multiple documents:
    {formatted_context[:40000]}
    
    Provide a comparative synthesis comparing and contrasting the viewpoints, facts, or methodologies described.
    Cite the documents by their titles in your synthesis.
    """
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CrossDocumentResponse,
                    system_instruction="You are a senior cognitive analyst. Synthesize information across multiple source texts, highlight alignments and contradictions, and write a beautiful markdown comparison."
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            if attempt == 2:
                raise e
            await asyncio.sleep(2 * (attempt + 1))
