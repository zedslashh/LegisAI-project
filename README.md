# LegisAI-Your personal AI assistant for legal tasks

A full-stack AI-powered legal document processing app that classifies, summarizes, and organizes legal PDFs. This system includes:

- A **Streamlit** frontend with navigation via `streamlit-option-menu`.
- A **FastAPI** backend for text extraction, classification, summarization, case metadata extraction, RAG-based QA, and more.

---

## 🚀 Features

### 📂 PDF Upload & Analysis
- Extracts text from PDFs (OCR fallback via Tesseract).
- Classifies documents into 15 legal categories (e.g., Domestic Violence, Property Dispute, etc.).
- Summarizes text using GPT-3.5 with multiple styles: `default`, `brief`, `detailed`, `structured`.
- Extracts structured case metadata (e.g., parties, judge, court, next hearing).

### 🧠 NLP & AI
- Embeds text using `text-embedding-ada-002` from OpenAI.
- Stores and retrieves embeddings using FAISS.
- RAG-based (Retrieval Augmented Generation) question answering via LangChain.

### 🏛️ Legal Case Management
- Search documents using natural language.
- View and sync case metadata.
- Track and retrieve related cases.

### 📊 Streamlit Frontend
- Built using `streamlit` and `streamlit-option-menu`.
- Interactive UI for uploading, browsing summaries, querying documents, and viewing case info.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit + streamlit-option-menu
- **Backend:** FastAPI + LangChain
- **NLP:** OpenAI GPT-3.5, HuggingFace BART zero-shot classifier
- **Embeddings:** OpenAI + FAISS
- **OCR & PDF Parsing:** pdfplumber, pytesseract, pdf2image
- **Calendar Integration:** icalendar, pytz
- **Email Notifications:** smtplib

---

## 📬 API Endpoints

- **POST** `/upload/`  
  Upload a PDF file and process it: extract text, classify, summarize, and store metadata.

- **GET** `/search/?query=your-question`  
  Perform semantic search across all uploaded documents using embeddings.

- **GET** `/rag_qa/?query=your-question`  
  Ask questions across uploaded documents using Retrieval-Augmented Generation (RAG) with LangChain.

- **GET** `/document/{doc_id}`  
  Retrieve the full text, summary, and category of a specific document by its ID.

- **GET** `/categories/`  
  List all predefined legal categories and the files stored under each.

- **GET** `/case/{case_id}/metadata`  
  Retrieve structured metadata for a specific case by ID (e.g., judge, court, hearing date).

- **POST** `/case/{case_id}/sync-court`  
  Simulate syncing case information with a court system.

---

## 🧪 Example Use Cases

- 📄 **Upload a legal PDF** (e.g., a judgment or petition) to:
  - Extract its content.
  - Generate a structured summary.
  - Classify the case into predefined categories (e.g., Land Dispute, Family Law).

- ❓ **Query your legal documents** with natural language:
  - Example: “What is the next hearing date for case #123?”
  - Example: “Find me a summary of the criminal offense cases.”

- 🔍 **Search for similar legal cases** using semantic embeddings:
  - Example: “Property dispute involving ancestral land” returns related documents.

- 🔗 **Explore related disputes**:
  - Detect and fetch cases marked as related in their metadata for cross-referencing.


## 🛠️ How to Run

Follow these steps to set up and execute the project on your local machine.

---

### 1. 📥 Clone the Repository

```bash
git clone https://github.com/zedslashh/LegisAI-project.git
cd LegisAI-project
```
### 2. 📤 Pull Latest Changes (Optional, if already cloned)

```bash
git pull origin main
```

### 3. 🧭 Navigate to the Project Directory

```bash
cd legal-analyzer
```

### 4. Execution: Run Backend & Frontend in Split Terminal

Use a code editor like **VS Code**, open a **split terminal**, and follow the steps below:

---

### ▶️ Terminal 1 — Run FastAPI Backend

```bash
uvicorn main:app --reload
```

### 🖥️ Terminal 2 — Run Streamlit Frontend

```bash
cd frontend
streamlit run app.py
```

