Here is the complete **Project Documentation** for your Hackathon submission.

You can copy and paste this directly into your GitHub `README.md` or print it as a PDF report for the judges.

---

# 📘 EduGraph: Domain-Specific AI Student Assistant

**Bridging the gap between static curriculum and interactive AI learning.**

---

## 1. Project Overview

**EduGraph** is a Retrieval-Augmented Generation (RAG) powered student assistant designed specifically for university environments. Unlike generic AI tools (ChatGPT, Gemini), EduGraph limits its knowledge base strictly to **teacher-approved curriculum documents**.

This ensures that every answer is **100% accurate to the syllabus**, fully cited with page numbers, and free from external hallucinations. It empowers students with interactive study tools while giving teachers real-time visibility into student learning gaps.

### **The "Why" (Problem Statement)**

* **Information Overload:** Students struggle to find specific concepts in hundreds of pages of static PDF notes.
* **The Trust Gap:** Generic AI models hallucinate or provide answers outside the university syllabus, making them unreliable for exam prep.
* **Educator Blind Spots:** Teachers upload materials to LMS platforms but have zero data on how students use them or where they struggle.
* **Passive Learning:** Students lack on-demand active recall tools (quizzes) tailored to their specific weekly units.

---

## 2. Key Features

### **For Students (The Learning Hub)**

* **🤖 Domain-Specific Chat:** Ask questions and get answers *only* from the uploaded Subject/Unit files.
* **🔍 "Page Peeking" Citations:** Every answer includes a direct verification link (e.g., `[Source: Unit 1 Notes.pdf, Page 12]`) to build trust.
* **🧠 Conversation Memory:** The AI remembers context (e.g., "Give me an example of *that*") for natural back-and-forth learning.
* **📝 AI Quiz Mode:** Instantly generates a 5-question Multiple Choice Quiz (MCQ) based on the selected unit for self-assessment.

### **For Teachers (The Command Center)**

* **📊 Analytics Dashboard:** A professional dashboard visualizing "Top Subjects," "Activity Trends," and "Recent Queries."
* **📂 Structured Knowledge Base:** Upload materials tagged by **Subject** and **Unit** (e.g., Physics > Unit 1) with auto-duplicate handling.
* **🗑️ Content Management:** Full control to add, update, or delete outdated materials instantly.

---

## 3. Technology Stack

| Component | Technology Used | Purpose |
| --- | --- | --- |
| **Backend** | Python (Flask) | API handling, Session logic, RAG pipeline orchestration. |
| **Database** | Supabase (PostgreSQL) | Auth, Chat Logs, and Vector Store (`pgvector`). |
| **AI Model** | Llama 3 (via Groq) | High-speed, low-latency LLM inference. |
| **Embeddings** | FastEmbed | Lightweight, local vector embedding generation. |
| **Frontend** | HTML5, JS, Chart.js | Responsive UI with real-time graphs and Markdown rendering. |
| **Orchestration** | LangChain | PDF loading, chunking, and text splitting. |

---

## 4. System Architecture

The project follows a **RAG (Retrieval-Augmented Generation)** architecture:

1. **Ingestion:** Teachers upload PDFs → Text is split into chunks → Embedded via FastEmbed → Stored in Supabase Vector Store.
2. **Retrieval:** Student asks a question → Query is embedded → Supabase performs a Similarity Search (Cosine Distance) *filtered by Subject/Unit*.
3. **Generation:** Retrieved context + Conversation History + User Query are sent to Llama 3 → AI generates a cited answer.
4. **Logging:** Interaction data (Subject, Unit, Query, Response) is logged to SQL for the Analytics Dashboard.

---

## 5. Installation & Setup Guide

### **Prerequisites**

* Python 3.11+
* Supabase Account (Free Tier)
* Groq API Key

### **Step 1: Clone & Install**

```bash
git clone https://github.com/your-username/edugraph.git
cd edugraph
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

```

### **Step 2: Environment Variables (`.env`)**

Create a `.env` file in the root directory:

```env
GROQ_API_KEY="your_groq_key"
SUPABASE_URL="your_supabase_url"
SUPABASE_KEY="your_supabase_anon_key"

```

### **Step 3: Database Setup (Supabase SQL)**

Run this SQL in your Supabase SQL Editor to set up the tables:

```sql
-- Enable Vector Extension
create extension vector;

-- Documents Table (Knowledge Base)
create table documents (
  id bigserial primary key,
  content text,
  metadata jsonb,
  embedding vector(384),
  user_id uuid
);

-- Chats Table (Analytics)
create table chats (
  id bigserial primary key,
  user_id uuid,
  message text,
  role text,
  subject text,
  unit text,
  created_at timestamptz default now()
);

-- Search Function (RPC)
create or replace function match_documents (
  query_embedding vector(384),
  match_threshold float,
  match_count int,
  filter jsonb
) returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
) language plpgsql stable as $$
begin
  return query
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where documents.metadata @> filter
  and 1 - (documents.embedding <=> query_embedding) > match_threshold
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;

```

### **Step 4: Run the App**

```bash
python app.py

```

* **Student Login:** Sign up with any email.
* **Teacher Login:** Sign up using the secure code: **`TEACHER_SECURE_2026`**.

---

## 6. Future Scope

* **"Content Gap" Alerts:** Automatically notify teachers when a student asks a question the database cannot answer.
* **Multi-Modal RAG:** Support for searching images and diagrams within PDFs.
* **LMS Integration:** Direct plugin for Moodle or Canvas.

---

**Built with ❤️ for the Hackathon 2026.**