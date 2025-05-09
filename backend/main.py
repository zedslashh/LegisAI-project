from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from pdf2image import convert_from_bytes
import pdfplumber
import openai
from transformers import pipeline
import faiss
import numpy as np
import os
import shutil
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pydantic import BaseModel
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import requests
from icalendar import Calendar, Event
import pytz
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create base directory for storing documents
BASE_DIR = "documents"
os.makedirs(BASE_DIR, exist_ok=True)

# Create category directories
category_labels = [
    "Domestic Violence", "Land Dispute", "Motor Vehicle Accident", "Theft",
    "Property Dispute", "Contract Dispute", "Family Law", "Employment",
    "Criminal Offense", "Civil Rights", "Environmental Law", "Taxation",
    "Intellectual Property", "Immigration", "Consumer Protection"
]

for category in category_labels:
    os.makedirs(os.path.join(BASE_DIR, category), exist_ok=True)

# In-memory storage
documents = []
summaries = []
categories = []
embeddings = []

# HuggingFace zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# FAISS index
embedding_dim = 1536  # OpenAI text-embedding-ada-002 dimension
faiss_index = faiss.IndexFlatL2(embedding_dim)

class CaseMetadata(BaseModel):
    case_number: str
    filing_date: str
    court: str
    judge: str
    parties: List[str]
    related_cases: List[str]
    status: str
    next_hearing: Optional[str]

# In-memory storage for case metadata
case_metadata: Dict[int, CaseMetadata] = {}

class NotificationSettings(BaseModel):
    email: str
    notify_before_hearing: int = 24  # hours
    notify_case_updates: bool = True
    notify_related_cases: bool = True

# Store notification settings
notification_settings: Dict[int, NotificationSettings] = {}

# Configure OpenAI API
openai.api_key = "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"
openai.api_base = "https://openrouter.ai/api/v1"

def extract_text_from_pdf(file_bytes):
    # Extract text directly from bytes without using temporary files
    try:
        # Use BytesIO instead of temporary file
        with io.BytesIO(file_bytes) as pdf_file:
            try:
                with pdfplumber.open(pdf_file) as pdf:
                    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                    if not text.strip():
                        # If digital extraction fails, try OCR
                        images = convert_from_bytes(file_bytes)
                        text = ""
                        for img in images:
                            text += pytesseract.image_to_string(img)
            except Exception as e:
                print(f"Error in PDF extraction: {str(e)}")
                # Fallback to OCR
                images = convert_from_bytes(file_bytes)
                text = ""
                for img in images:
                    text += pytesseract.image_to_string(img)
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        text = ""
    return text

def summarize_text(text, style="default"):
    try:
        if style == "detailed":
            prompt = f"""Provide a detailed summary of the following legal case in 8-10 sentences. Include:
1. Key facts and events
2. Main legal issues
3. Arguments presented
4. Court's decision
5. Legal implications

Text: {text[:3000]}"""
        elif style == "brief":
            prompt = f"Summarize the following legal case in 3 sentences, focusing on the main issue and outcome:\n\n{text[:3000]}"
        elif style == "structured":
            prompt = f"""Analyze the following legal case and provide a structured summary with these sections:
1. Case Overview
2. Key Facts
3. Legal Issues
4. Decision
5. Impact

Text: {text[:3000]}"""
        else:  # default
            prompt = f"Summarize the following legal case in 5 sentences:\n\n{text[:3000]}"

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500 if style == "detailed" else 200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in summarization: {str(e)}")
        return "Error generating summary"

def get_embedding(text):
    try:
        response = openai.Embedding.create(
            input=text[:3000],
            model="text-embedding-ada-002"
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    except Exception as e:
        print(f"Error in embedding: {str(e)}")
        return np.zeros(embedding_dim, dtype=np.float32)

def classify_text(text):
    try:
        result = classifier(text[:1000], category_labels)
        return result['labels'][0]
    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return "Unknown"

def extract_case_metadata(text: str) -> CaseMetadata:
    try:
        prompt = f"""Extract the following information from this legal case:
1. Case number
2. Filing date
3. Court name
4. Judge name
5. Parties involved
6. Related case numbers
7. Current status
8. Next hearing date (if any)

Format the response as a JSON object with these fields.
Text: {text[:3000]}"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        
        # Parse the response and create metadata
        metadata_text = response.choices[0].message.content.strip()
        # Basic parsing (you might want to add more robust parsing)
        return CaseMetadata(
            case_number=metadata_text.split('"case_number": "')[1].split('"')[0] if '"case_number": "' in metadata_text else "Unknown",
            filing_date=metadata_text.split('"filing_date": "')[1].split('"')[0] if '"filing_date": "' in metadata_text else "Unknown",
            court=metadata_text.split('"court": "')[1].split('"')[0] if '"court": "' in metadata_text else "Unknown",
            judge=metadata_text.split('"judge": "')[1].split('"')[0] if '"judge": "' in metadata_text else "Unknown",
            parties=metadata_text.split('"parties": [')[1].split(']')[0].replace('"', '').split(',') if '"parties": [' in metadata_text else [],
            related_cases=metadata_text.split('"related_cases": [')[1].split(']')[0].replace('"', '').split(',') if '"related_cases": [' in metadata_text else [],
            status=metadata_text.split('"status": "')[1].split('"')[0] if '"status": "' in metadata_text else "Unknown",
            next_hearing=metadata_text.split('"next_hearing": "')[1].split('"')[0] if '"next_hearing": "' in metadata_text else None
        )
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
        return CaseMetadata(
            case_number="Unknown",
            filing_date="Unknown",
            court="Unknown",
            judge="Unknown",
            parties=[],
            related_cases=[],
            status="Unknown",
            next_hearing=None
        )

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...), summary_style: str = "default"):
    try:
        file_bytes = await file.read()
        text = extract_text_from_pdf(file_bytes)
        if not text.strip():
            return {"error": "Could not extract text from PDF"}
            
        summary = summarize_text(text, summary_style)
        if summary == "Error generating summary":
            return {"error": "Failed to generate summary"}
            
        category = classify_text(text)
        embedding = get_embedding(text)
        
        # Extract case metadata
        metadata = extract_case_metadata(text)
        
        # Store in memory
        idx = len(documents)
        documents.append(text)
        summaries.append(summary)
        categories.append(category)
        embeddings.append(embedding)
        case_metadata[idx] = metadata
        faiss_index.add(np.array([embedding]))

        # Save file to category folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        category_path = os.path.join(BASE_DIR, category)
        file_path = os.path.join(category_path, filename)
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        
        return {
            "id": idx,
            "summary": summary,
            "category": category,
            "filename": filename,
            "file_path": file_path,
            "summary_style": summary_style,
            "metadata": metadata.dict()
        }
    except Exception as e:
        print(f"Error in upload: {str(e)}")
        return {"error": str(e)}

@app.get("/search/")
def search(query: str):
    try:
        query_emb = get_embedding(query)
        D, I = faiss_index.search(np.array([query_emb]), k=5)
        results = []
        for idx in I[0]:
            if idx < len(documents):
                results.append({
                    "id": idx,
                    "summary": summaries[idx],
                    "category": categories[idx]
                })
        return results
    except Exception as e:
        print(f"Error in search: {str(e)}")
        return []

@app.get("/document/{doc_id}")
def get_document(doc_id: int):
    try:
        if 0 <= doc_id < len(documents):
            return {
                "text": documents[doc_id],
                "summary": summaries[doc_id],
                "category": categories[doc_id]
            }
        return {"error": "Document not found"}
    except Exception as e:
        print(f"Error in get_document: {str(e)}")
        return {"error": str(e)}

@app.get("/categories/")
def get_categories():
    categories_info = []
    for category in category_labels:
        category_path = os.path.join(BASE_DIR, category)
        files = os.listdir(category_path)
        categories_info.append({
            "name": category,
            "count": len(files),
            "files": [f for f in files if f.endswith('.pdf')]
        })
    return categories_info

@app.get("/category/{category_name}")
def get_category_files(category_name: str):
    if category_name not in category_labels:
        return {"error": "Invalid category"}
    
    category_path = os.path.join(BASE_DIR, category_name)
    files = os.listdir(category_path)
    return {
        "category": category_name,
        "files": [f for f in files if f.endswith('.pdf')]
    }

@app.get("/case/{case_id}/metadata")
def get_case_metadata(case_id: int):
    if case_id not in case_metadata:
        raise HTTPException(status_code=404, detail="Case not found")
    return case_metadata[case_id]

@app.get("/cases/related/{case_id}")
def get_related_cases(case_id: int):
    if case_id not in case_metadata:
        raise HTTPException(status_code=404, detail="Case not found")
    
    related_cases = []
    for idx, metadata in case_metadata.items():
        if case_id != idx and (
            case_metadata[case_id].case_number in metadata.related_cases or
            metadata.case_number in case_metadata[case_id].related_cases
        ):
            related_cases.append({
                "id": idx,
                "case_number": metadata.case_number,
                "summary": summaries[idx],
                "category": categories[idx]
            })
    return related_cases

@app.get("/cases/timeline")
def get_case_timeline():
    timeline = []
    for idx, metadata in case_metadata.items():
        if metadata.filing_date != "Unknown":
            timeline.append({
                "id": idx,
                "case_number": metadata.case_number,
                "filing_date": metadata.filing_date,
                "court": metadata.court,
                "status": metadata.status,
                "next_hearing": metadata.next_hearing,
                "category": categories[idx]
            })
    return sorted(timeline, key=lambda x: x["filing_date"])

@app.get("/cases/upcoming")
def get_upcoming_cases():
    upcoming = []
    for idx, metadata in case_metadata.items():
        if metadata.next_hearing:
            upcoming.append({
                "id": idx,
                "case_number": metadata.case_number,
                "next_hearing": metadata.next_hearing,
                "court": metadata.court,
                "category": categories[idx]
            })
    return sorted(upcoming, key=lambda x: x["next_hearing"])

def send_email_notification(to_email: str, subject: str, body: str):
    try:
        # Configure your email settings
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = os.getenv("EMAIL_USER")
        sender_password = os.getenv("EMAIL_PASSWORD")

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'html'))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
    except Exception as e:
        print(f"Error sending email: {str(e)}")

def create_calendar_event(case_number: str, hearing_date: str, court: str, description: str) -> str:
    cal = Calendar()
    event = Event()
    event.add('summary', f'Court Hearing: {case_number}')
    event.add('dtstart', datetime.strptime(hearing_date, "%Y-%m-%d %H:%M"))
    event.add('dtend', datetime.strptime(hearing_date, "%Y-%m-%d %H:%M") + timedelta(hours=2))
    event.add('location', court)
    event.add('description', description)
    
    cal.add_component(event)
    
    # Save to file
    filename = f"hearing_{case_number.replace(' ', '_')}.ics"
    with open(filename, 'wb') as f:
        f.write(cal.to_ical())
    return filename

def notify_upcoming_hearing(background_tasks: BackgroundTasks, case_id: int):
    if case_id in case_metadata and case_id in notification_settings:
        case = case_metadata[case_id]
        settings = notification_settings[case_id]
        
        if case.next_hearing:
            hearing_time = datetime.strptime(case.next_hearing, "%Y-%m-%d %H:%M")
            notification_time = hearing_time - timedelta(hours=settings.notify_before_hearing)
            
            if datetime.now() <= notification_time:
                background_tasks.add_task(
                    send_email_notification,
                    settings.email,
                    f"Upcoming Hearing: {case.case_number}",
                    f"""
                    <h2>Upcoming Court Hearing</h2>
                    <p><strong>Case Number:</strong> {case.case_number}</p>
                    <p><strong>Court:</strong> {case.court}</p>
                    <p><strong>Date:</strong> {case.next_hearing}</p>
                    <p><strong>Judge:</strong> {case.judge}</p>
                    <p><strong>Status:</strong> {case.status}</p>
                    """
                )

@app.post("/case/{case_id}/notifications")
def set_notification_settings(case_id: int, settings: NotificationSettings):
    if case_id not in case_metadata:
        raise HTTPException(status_code=404, detail="Case not found")
    
    notification_settings[case_id] = settings
    return {"message": "Notification settings updated"}

@app.get("/case/{case_id}/calendar")
def get_calendar_event(case_id: int):
    if case_id not in case_metadata:
        raise HTTPException(status_code=404, detail="Case not found")
    
    case = case_metadata[case_id]
    if not case.next_hearing:
        raise HTTPException(status_code=400, detail="No upcoming hearing")
    
    description = f"""
    Case Number: {case.case_number}
    Court: {case.court}
    Judge: {case.judge}
    Status: {case.status}
    """
    
    filename = create_calendar_event(
        case.case_number,
        case.next_hearing,
        case.court,
        description
    )
    
    return {"calendar_file": filename}

# Integration with legal research APIs
def search_legal_precedents(case_text: str) -> List[Dict]:
    try:
        # Example integration with a legal research API
        # Replace with actual API endpoint and authentication
        api_url = "https://api.legal-research.com/search"
        headers = {
            "Authorization": f"Bearer {os.getenv('LEGAL_RESEARCH_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            api_url,
            json={"text": case_text[:1000]},
            headers=headers
        )
        
        if response.ok:
            return response.json()["results"]
        return []
    except Exception as e:
        print(f"Error searching legal precedents: {str(e)}")
        return []

@app.get("/case/{case_id}/precedents")
def get_legal_precedents(case_id: int):
    if case_id not in documents:
        raise HTTPException(status_code=404, detail="Case not found")
    
    precedents = search_legal_precedents(documents[case_id])
    return {"precedents": precedents}

# Integration with court management system
def update_court_system(case_data: Dict):
    try:
        # Example integration with court management system
        # Replace with actual API endpoint and authentication
        api_url = "https://api.court-system.com/cases"
        headers = {
            "Authorization": f"Bearer {os.getenv('COURT_SYSTEM_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            api_url,
            json=case_data,
            headers=headers
        )
        
        return response.ok
    except Exception as e:
        print(f"Error updating court system: {str(e)}")
        return False

@app.post("/case/{case_id}/sync-court")
def sync_with_court_system(case_id: int):
    if case_id not in case_metadata:
        raise HTTPException(status_code=404, detail="Case not found")
    
    case = case_metadata[case_id]
    case_data = {
        "case_number": case.case_number,
        "filing_date": case.filing_date,
        "court": case.court,
        "judge": case.judge,
        "status": case.status,
        "next_hearing": case.next_hearing,
        "parties": case.parties
    }
    
    if update_court_system(case_data):
        return {"message": "Case synchronized with court system"}
    else:
        raise HTTPException(status_code=500, detail="Failed to sync with court system")

@app.get("/rag_qa/")
def rag_qa(query: str = Query(...)):
    if not documents:
        return {"answer": "No documents uploaded yet."}
        
    # Use direct OpenAI API for simpler implementation
    try:
        # Find the most relevant document using embeddings
        query_emb = get_embedding(query)
        D, I = faiss_index.search(np.array([query_emb]), k=3)
        
        # Combine the top documents
        context = ""
        for idx in I[0]:
            if idx < len(documents):
                # Add a portion of each document
                context += documents[idx][:1000] + "\n\n"
        
        prompt = f"""Answer the following question based only on the provided context:

Context:
{context[:3000]}

Question: {query}

Answer:"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        answer = response.choices[0].message.content.strip()
        
        return {"answer": answer, "context": context[:500] + "..." if len(context) > 500 else context}
    except Exception as e:
        print(f"Error in RAG QA: {str(e)}")
        return {"answer": f"Error generating answer: {str(e)}", "context": ""}

@app.get("/case_features/")
def case_features():
    features = []
    for idx, meta in case_metadata.items():
        features.append({
            "case_number": meta.case_number,
            "filing_date": meta.filing_date,
            "court": meta.court,
            "judge": meta.judge,
            "parties": ", ".join(meta.parties) if meta.parties else None,
            "related_cases": ", ".join(meta.related_cases) if meta.related_cases else None,
            "status": meta.status,
            "next_hearing": meta.next_hearing
        })
    return features

# Run the server when the script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)