# RAG Chatbot Portfolio Project

A full-stack AI chatbot that allows users to upload PDF documents and ask questions about them. 
It uses Retrieval-Augmented Generation (RAG) to provide accurate, document-grounded answers without hallucinations.

## Tech Stack
* **Frontend:** HTML, CSS, JavaScript
* **Backend:** Python, Flask
* **Database:** Supabase (Vector Store & Auth)
* **AI Model:** Llama 3 via Groq API
* **Embeddings:** Hugging Face (Sentence Transformers)

## How to Run Locally
1.  Clone the repository.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Create a `.env` file with your API keys.
4.  Run `python app.py`.