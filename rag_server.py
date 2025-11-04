# rag_server.py
# This server runs in a separate terminal and environment (rag_env) to isolate 
# the heavy sentence-transformers/PDF dependencies from the main Gradio app.

import os
import json
import numpy as np
from pathlib import Path
import re
import fitz # PyMuPDF
from fastapi import FastAPI, HTTPException
import uvicorn


# --- Helper to extract the best sentence ---
def get_best_answer_sentence(query, context_lines):
    """
    Finds the most relevant sentence in the context that answers a 'what is' query.
    """
    query_lower = query.lower()
    best_sentence = context_lines[0] # Default to the first line
    max_score = -1

    # Search for the best sentence based on query keywords
    for sentence in context_lines:
        score = 0
        s_lower = sentence.lower()
        
        # Boost score if the sentence starts with the definition (e.g., "The CTR is...")
        if query_lower in s_lower or s_lower.startswith(("the", "a", "an")):
            score += 2
        
        # High boost for sentences that explicitly define the target term
        if "ctr" in query_lower and ("cardiothoracic ratio" in s_lower or "ratio" in s_lower):
            score += 5
        if "hybridgnet" in query.lower() and ("architecture" in s_lower or "neural network" in s_lower):
            score += 5
        
        # Check if the sentence is long enough to be an answer, not just a label
        if len(s_lower.split()) > 10:
            score += 1
            
        if score > max_score:
            max_score = score
            best_sentence = sentence
            
    return best_sentence

# -------------------- GENERATION LOGIC (Refined) --------------------


# Import Sentence Transformers components (must be installed in rag_env)
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    print("FATAL ERROR: sentence-transformers is not installed in this environment. Please check 'rag_env'.")
    raise

# -------------------- RAG CONFIG --------------------
# Define paths relative to the server script
PAPERS_DIR = "/Users/joshua/College/Chest-x-ray-HybridGNet-Segmentation/papers" 
CHUNK_SIZE = 500
OVERLAP = 100

# -------------------- GLOBAL RAG STATE --------------------
# This state persists while the FastAPI server is running


# -------------------- RAG CORE UTILITIES --------------------
DOC_STATE = {'embedder': None, 'doc_embeds': None, 'documents': [], 'session_context': ""}

def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}") 
        return ""

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Cleans and chunks text with specified overlap."""
    if not text.strip(): return []
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start >= len(text): break
    return chunks

def retrieve_info(query: str, top_k=3):
    """Retrieves top_k most relevant chunks using cosine similarity on embeddings."""
    embedder, doc_embeds, documents = DOC_STATE['embedder'], DOC_STATE['doc_embeds'], DOC_STATE['documents']
    
    if embedder is None or doc_embeds is None or len(documents) == 0:
        raise ValueError("RAG state is empty. Initialization failed or documents are missing.")
    
    query_embed = embedder.encode(query, convert_to_numpy=True)
    scores = util.cos_sim(query_embed, doc_embeds)[0]
    top_indices = scores.topk(min(top_k, len(documents))).indices.tolist()
    
    return [documents[i] for i in top_indices]

# -------------------- RAG INITIALIZATION --------------------

def initialize_rag():
    """Initializes the Sentence Transformer and loads all documents once on startup."""
    print("Starting RAG system initialization...")
    
    # === EMBEDDING MODEL ===
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    documents = []
    
    # 1. Add static knowledge base
    documents.extend([
        "Cardiomegaly is a medical condition in which the heart is enlarged. It can be detected on a chest X-ray as an increased cardiothoracic ratio (CTR > 0.5).",
        "A normal cardiothoracic ratio (CTR) is typically less than 0.5 in adults on a PA chest X-ray.",
        "Disclaimer: This information is for educational purposes only and is not a substitute for professional medical advice."
    ])
    
    # 2. Load all PDFs from initial directory
    papers_path = Path(PAPERS_DIR)
    if papers_path.exists():
        pdf_files = list(papers_path.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files. Indexing...")
        
        for pdf_file in pdf_files:
            raw_text = extract_text_from_pdf(pdf_file.as_posix()) # Use as_posix() for cross-platform path safety
            chunks = chunk_text(raw_text)
            for i, chunk in enumerate(chunks):
                metadata = f"[Source: {pdf_file.name}]"
                documents.append(f"{chunk}\n\n{metadata}")

    # 3. Encode documents
    if documents:
        doc_embeds = embedder.encode(documents, show_progress_bar=False, convert_to_numpy=True)
    else:
        doc_embeds = np.array([])

    # Update global state
    DOC_STATE['embedder'] = embedder
    DOC_STATE['doc_embeds'] = doc_embeds
    DOC_STATE['documents'] = documents
    
    print(f"RAG system fully loaded with {len(documents)} document chunks. Server is ready.")

# -------------------- GENERATION LOGIC --------------------
# --- Helper to extract the best sentence (Slightly improved) ---
def get_best_answer_sentence(query, context_lines):
    """
    Finds the most relevant sentence in the context that answers a 'what is' query.
    """
    query_lower = query.lower()
    best_sentence = context_lines[0] # Default to the first line
    max_score = -1

    # Search for the best sentence based on query keywords
    for sentence in context_lines:
        score = 0
        s_lower = sentence.lower()
        
        # Boost score if the sentence explicitly contains a definition (is, means, refers to)
        if " is a" in s_lower or " is an" in s_lower or " refers to" in s_lower:
             score += 3
        
        # High boost for sentences that explicitly define the target term
        if "ctr" in query_lower and ("cardiothoracic ratio" in s_lower or "ratio" in s_lower):
            score += 5
        if "hybridgnet" in query.lower() and ("architecture" in s_lower or "neural network" in s_lower):
            score += 5
        
        # General boost for keyword matches
        if query_lower in s_lower:
            score += 2
            
        if score > max_score:
            max_score = score
            best_sentence = sentence
            
    return best_sentence
def generate_answer(query: str):
    """Retrieves context, synthesizes answer, and includes dynamic session info only when relevant."""
    
    if DOC_STATE['embedder'] is None:
        raise ValueError("RAG state is not initialized.")
    
    session_context = DOC_STATE.get('session_context', "")
    advisory_note = ""
    personalized_note = ""
    import re

    # --- Determine if query is about the patient ---
    lower_q = query.lower().strip()

    # Define regex patterns that indicate a *personal* question
    personal_indicators = [
        # Personal pronouns / ownership
        r"\bmy\b", r"\bme\b", r"\bmine\b", r"\bmyself\b",

        # Report / scan / test references
        r"\b(result|results|report|scan|x[- ]?ray|image|test|finding|reading)\b",

        # Diagnostic intent
        r"\b(diagnosis|interpret|meaning|significance|severity|outcome)\b",

        # Risk or health condition queries
        r"\b(condition|risk|disease|problem|abnormal|concern)\b",

        # Medical interaction phrasing
        r"\b(doctor|physician|consult|recommendation|next step)\b",

        # Natural question phrasing
        r"what does (it|this) mean", r"how bad", r"how serious", r"is it normal",
        r"should i worry", r"am i", r"do i have", r"does this indicate"
    ]

    # Define explicit exceptions for general educational questions
    general_exceptions = [
        "what is cardiomegaly",
        "define cardiomegaly",
        "explain cardiomegaly",
        "what is cdr",
        "what is ctr",
        "what is the heart ratio"
    ]

    # Determine match using regex (not substring)
    user_is_asking_about_self = any(re.search(p, lower_q) for p in personal_indicators)

    # Override if query matches any general exception
    if any(exc in lower_q for exc in general_exceptions):
        user_is_asking_about_self = False

        
    # --- Parse context only if query is personal ---
    if user_is_asking_about_self and session_context:
        personalized_note = f"\n\n[Patient Context]: {session_context}\n\n"

        import re
        likelihood_match = re.search(r"likelihood[: ]+([\d.]+)%", session_context.lower())
        cdr_match = re.search(r"cdr[: ]+([\d.]+)", session_context.lower())
        hr_match = re.search(r"heart rate[: ]+([\d.]+)", session_context.lower())

        likelihood = float(likelihood_match.group(1)) if likelihood_match else None
        cdr = float(cdr_match.group(1)) if cdr_match else None
        heart_rate = float(hr_match.group(1)) if hr_match else None

        # --- Personalized clinical interpretation logic ---
        if likelihood is not None or cdr is not None:
            if (likelihood and likelihood > 80) and (cdr and cdr > 0.55):
                advisory_note = (
                    "🩺 **High Clinical Concern:** Both the cardiothoracic ratio (CDR) and model-predicted "
                    "likelihood are elevated. This strongly suggests possible cardiomegaly. "
                    "A cardiology consultation and echocardiogram are recommended.\n\n"
                )
            elif (cdr and cdr > 0.55) and (likelihood and likelihood < 50):
                advisory_note = (
                    "⚖️ **Mixed Findings:** Your CDR is high (>0.55) but the model predicts a low likelihood. "
                    "This may indicate benign enlargement or imaging artifacts. "
                    "Follow-up imaging or physician review is advised.\n\n"
                )
            elif (cdr and cdr <= 0.5) and (likelihood and likelihood > 75):
                advisory_note = (
                    "🧩 **Model Detected Anomaly:** The CDR is normal (<0.50) but model confidence is high. "
                    "Possible early or shape-based cardiac changes. Recommend further clinical review.\n\n"
                )
            elif (likelihood and 60 <= likelihood <= 80) or (cdr and 0.50 <= cdr <= 0.55):
                advisory_note = (
                     "🩺 **High Clinical Concern:** Both the cardiothoracic ratio (CDR) and model-predicted "
                    "likelihood are elevated. This strongly suggests possible cardiomegaly. "
                    "A cardiology consultation and echocardiogram are recommended.\n\n"
                )
            else:
                advisory_note = (
                    "✅ **Normal Range:** Both your CDR and cardiomegaly likelihood are within normal limits. "
                    "No immediate concern indicated, though routine monitoring is recommended.\n\n"
                )

        # --- Add heart rate note ---
        if heart_rate is not None:
            if heart_rate < 60:
                advisory_note += f"❤️ **Heart Rate:** {heart_rate:.0f} bpm — *Bradycardia (Low)*.\n\n"
            elif heart_rate > 100:
                advisory_note += f"❤️ **Heart Rate:** {heart_rate:.0f} bpm — *Tachycardia (High)*.\n\n"
            else:
                advisory_note += f"❤️ **Heart Rate:** {heart_rate:.0f} bpm — *Normal Range*.\n\n"

    # --- Retrieve relevant research context ---
    if not user_is_asking_about_self:
        retrieved_chunks = retrieve_info(query)
        combined_text = "\n\n---\n\n".join(retrieved_chunks)
        context_lines = [line for line in combined_text.split('\n')
                         if line.strip() and not line.startswith('[Source:')]

        if not context_lines:
            core_sentence = (
                "I couldn’t find a direct research-based answer, "
                "but your findings should be reviewed by a healthcare provider."
            )
        else:
            core_sentence = get_best_answer_sentence(query, context_lines)

        answer_text = f"**{core_sentence}**"
    else:
        # Personalized query → use clinical interpretation only
        answer_text = advisory_note
        retrieved_chunks = []

    # --- Extract sources ---
    import re
    source_names = set()
    for chunk in retrieved_chunks:
        match = re.search(r'\[Source: (.*?)\]', chunk)
        if match:
            source_names.add(match.group(1))

    return {
        "answer": answer_text,
        "sources": list(source_names)
    }



app = FastAPI(
    title="HybridGNet RAG Worker",
    description="Dedicated server for semantic search against indexed research papers.",
    on_startup=[initialize_rag] # Initialization happens automatically when server starts
)


@app.post("/update_context")
def update_context(payload: dict):
    """
    Updates the temporary context in the RAG server with new patient results
    so the chatbot can answer personalized questions.
    """
    context = payload.get("context", "")
    if not context.strip():
        raise HTTPException(status_code=400, detail="No valid context provided.")
    
    DOC_STATE['session_context'] = context.strip()
    return {"status": "success", "message": "Context updated successfully."}
# -------------------- API ENDPOINT --------------------

@app.post("/clear_context")
def clear_context():
    DOC_STATE['session_context'] = ""
    return {"status": "success", "message": "Context cleared."}

@app.get("/query", response_model=dict)
def api_query(query: str):
    """Endpoint for querying the RAG model."""
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required.")
        
    try:
        response = generate_answer(query)
        # Return the structured data which the client expects
        return {"status": "success", "data": response}
    except Exception as e:
        # Catch any unexpected runtime errors
        raise HTTPException(status_code=500, detail=f"Internal Server Error during query: {str(e)}")

if __name__ == "__main__":
    # Start the server on a dedicated port (e.g., 8000)
    print("\n--- RAG SERVER STARTING ---")
    print(f"Indexing papers from: {PAPERS_DIR}")
    # The server will run the initialize_rag function automatically when it starts.
    uvicorn.run(app, host="127.0.0.1", port=8000)