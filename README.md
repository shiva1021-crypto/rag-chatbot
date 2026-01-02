Here is a professional, ready-to-use `README.md` file for your GitHub repository. It covers all the new features, setup instructions, and the architecture of your updated project.

Copy the content below and save it as `README.md` in your project folder.

---

```markdown
# 🎓 EduGraph: Domain-Specific Student Assistant RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Framework-Flask-green)
![Supabase](https://img.shields.io/badge/Database-Supabase%20%2B%20pgvector-emerald)
![AI](https://img.shields.io/badge/AI-Llama%203%20(Groq)-orange)

**EduGraph** is a full-stack Retrieval-Augmented Generation (RAG) platform designed for educational institutions. Unlike generic AI chatbots, it allows teachers to create a **curated knowledge base** organized by Subjects and Units. Students can then chat with an AI Tutor that answers questions strictly grounded in the specific syllabus notes uploaded by the teacher.

---

## 🚀 Key Features

### 👨‍🏫 Teacher Portal (Admin)
* **Secure Access:** "Gatekeeper" login system using a secret Teacher Access Code.
* **Syllabus Management:** Upload PDF documents tagged by **Subject** (e.g., Physics) and **Unit** (e.g., Module 1).
* **Library View:** View all uploaded files organized hierarchically.

### 🎓 Student Portal
* **Context-Aware Study:** Select specific Subjects and Units to focus the AI's knowledge.
* **RAG Chatbot:** Ask questions and get answers based *only* on the selected unit's material.
* **Source Citations:** Every answer cites the specific source file used.
* **Zero Hallucinations:** The AI refuses to answer questions outside the uploaded syllabus.

---

## 🛠️ Tech Stack

* **Backend:** Python (Flask)
* **Database:** Supabase (PostgreSQL)
* **Vector Search:** `pgvector` (Cosine Similarity with Metadata Filtering)
* **LLM:** Llama 3.1 8B Instant (via Groq API)
* **Embeddings:** `BAAI/bge-small-en-v1.5` (via FastEmbed)
* **Frontend:** HTML5, CSS3, Vanilla JavaScript
* **Orchestration:** LangChain

---

## ⚙️ Installation & Setup

### 1. Prerequisites
* **Python 3.11** (Required for FastEmbed compatibility).
* A **Supabase** project (Free tier).
* A **Groq** API Key (Free tier).

### 2. Clone the Repository
```bash
git clone [https://github.com/your-username/EduGraph.git](https://github.com/your-username/EduGraph.git)
cd EduGraph

```

### 3. Set Up Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

```

### 4. Install Dependencies

```bash
pip install -r requirements.txt

```

### 5. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key

```

---

## 🗄️ Database Setup (Supabase)

Go to your Supabase **SQL Editor** and run the following commands to set up the schema and permissions:

```sql
-- 1. Enable Vector Extension
create extension if not exists vector;

-- 2. Create Documents Table (Vector Store)
create table documents (
  id bigserial primary key,
  content text,
  metadata jsonb,
  embedding vector(384), -- Matching BAAI/bge-small-en-v1.5 dimensions
  user_id uuid
);

-- 3. Create Profiles Table (Role Management)
create table profiles (
  id uuid references auth.users not null primary key,
  email text,
  role text default 'student'
);

-- 4. Create Match Function (For Vector Search)
create or replace function match_documents (
  query_embedding vector(384),
  match_threshold float,
  match_count int,
  filter jsonb
)
returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where 1 - (documents.embedding <=> query_embedding) > match_threshold
  and documents.metadata @> filter
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- 5. Disable RLS (To allow Python backend full access)
alter table documents disable row level security;
alter table profiles disable row level security;

```

---

## 🏃‍♂️ Usage Guide

### Starting the Server

```bash
python app.py

```

Access the app at `http://127.0.0.1:5000`.

### Roles & Login

1. **Student:**
* Click "Sign Up".
* Role defaults to Student.
* Log in to access the study portal.


2. **Teacher (Admin):**
* Click "Teacher" on the Sign In/Sign Up page.
* Enter the **Teacher Access Code** (Default: `TEACHER_SECURE_2026`).
* *Note: You can change this code in `app.py` line 34.*



### Uploading Syllabus

1. Log in as **Teacher**.
2. Enter **Subject Name** (e.g., "Data Structures") and **Unit** (e.g., "Unit 1").
3. Upload the PDF.

### Chatting

1. Log in as **Student**.
2. Select the **Subject** and **Unit** from the sidebar.
3. Ask a question! The AI will answer based on the uploaded file.

---

## 📂 Project Structure

```
EduGraph/
├── app.py                  # Main Flask backend & RAG logic
├── requirements.txt        # Python dependencies
├── .env                    # API keys (Not on GitHub)
├── templates/
│   ├── index.html          # Main Dashboard (Student & Teacher views)
│   └── login.html          # Authentication & Role Selection
└── temp_uploads/           # Temporary storage for processing PDFs

```

## 🤝 Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first.

## 📜 License

This project is open-source and available under the MIT License.

```

```