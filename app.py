import os
import json
from collections import Counter
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from groq import Groq
from supabase import create_client, Client

# 1. Load Environment Variables
load_dotenv()

app = Flask(__name__)

# --- CRITICAL FIX 1: Stable Session Management ---
# Uses a fixed key from .env so users stay logged in after server restarts.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_fallback_key_do_not_use_in_prod")

# --- Configuration ---
BASE_UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(BASE_UPLOAD_FOLDER, exist_ok=True)

# --- CRITICAL FIX 2: Secure Admin Token ---
# Loaded from .env to prevent hardcoded secrets in source code.
ADMIN_ACCESS_TOKEN = os.environ.get("ADMIN_ACCESS_CODE")
if not ADMIN_ACCESS_TOKEN:
    print("WARNING: ADMIN_ACCESS_CODE is not set in your .env file!")

# --- Init Clients ---
# 1. Groq
client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# 2. Supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# 3. Embeddings (Lightweight for Free Tier)
embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# =========================================================
#  FRONTEND & AUTH ROUTES
# =========================================================

@app.route('/get_missed_queries', methods=['GET'])
def get_missed_queries():
    if session.get('role') != 'admin': return jsonify({"error": "Unauthorized"}), 403
    
    try:
        # Fetch last 5 missed queries
        response = supabase.table('missed_queries') \
            .select("*") \
            .order('created_at', desc=True) \
            .limit(5) \
            .execute()
            
        return jsonify(response.data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/')
def index():
    if 'user_id' not in session: return render_template('login.html')
    return render_template('index.html')

@app.route('/login_page')
def login_page(): return render_template('login.html')

@app.route('/auth/signup', methods=['POST'])
def signup():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'student')
    admin_code = data.get('admin_code')

    # --- SECURITY CHECK ---
    if role == 'admin':
        if admin_code != ADMIN_ACCESS_TOKEN:
            return jsonify({"error": "Invalid Admin Access Code! You are not authorized."}), 403
    
    try:
        response = supabase.auth.sign_up({"email": email, "password": password})
        
        if response.user:
            if role == 'admin':
                supabase.table('profiles').update({'role': 'admin'}).eq('id', response.user.id).execute()
            return jsonify({"message": f"Signup successful as {role}!"})
        
        return jsonify({"error": "Signup failed"}), 400
    except Exception as e: 
        return jsonify({"error": str(e)}), 400

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.json
    try:
        # 1. Authenticate with Supabase
        response = supabase.auth.sign_in_with_password({"email": data.get('email'), "password": data.get('password')})
        user = response.user
        
        # 2. Fetch Role from 'profiles' table
        profile_res = supabase.table('profiles').select('role').eq('id', user.id).execute()
        user_role = profile_res.data[0]['role'] if profile_res.data else 'student'

        # 3. Save Session
        session['user_id'] = user.id
        session['email'] = user.email
        session['role'] = user_role 

        return jsonify({"message": "Login successful", "redirect": "/", "role": user_role})
    except Exception as e: 
        print(f"Login Error: {e}")
        return jsonify({"error": "Invalid credentials"}), 401

@app.route('/auth/logout')
def logout():
    session.clear()
    return redirect('/login_page')

@app.route('/get_user_info', methods=['GET'])
def get_user_info():
    if 'user_id' not in session: return jsonify({"error": "Unauthorized"}), 401
    return jsonify({
        "email": session.get('email'),
        "user_id": session.get('user_id'),
        "initial": session.get('email')[0].upper() if session.get('email') else "?"
    })

# =========================================================
#  DASHBOARD & ANALYTICS (ADMIN)
# =========================================================

@app.route('/dashboard_stats', methods=['GET'])
def dashboard_stats():
    if session.get('role') != 'admin':
        return jsonify({"error": "Unauthorized"}), 403

    try:
        # 1. Fetch all student queries (role='user')
        response = supabase.table('chats').select("*") \
            .eq('role', 'user') \
            .order('created_at', desc=True) \
            .limit(1000).execute()
        
        data = response.data
        if not data: return jsonify({"total_queries": 0})

        # 2. Compute Metrics
        subjects = [row.get('subject', 'Unknown') for row in data if row.get('subject')]
        subject_counts = dict(Counter(subjects))

        dates = [row['created_at'].split('T')[0] for row in data]
        date_counts = dict(Counter(dates))
        sorted_dates = sorted(date_counts.keys())
        
        activity_trend = {
            "labels": sorted_dates,
            "data": [date_counts[d] for d in sorted_dates]
        }
        
        unique_students = len(set(row['user_id'] for row in data))

        return jsonify({
            "total_queries": len(data),
            "active_students": unique_students,
            "subject_distribution": subject_counts,
            "activity_trend": activity_trend,
            "recent_queries": [row['message'] for row in data[:5]]
        })

    except Exception as e:
        print(f"Analytics Error: {e}")
        return jsonify({"error": str(e)}), 500

# =========================================================
#  CORE RAG FEATURES (UPLOAD, LIBRARY, CHAT)
# =========================================================

@app.route('/upload', methods=['POST'])
def upload_file():
    if session.get('role') != 'admin':
        return jsonify({"error": "Unauthorized"}), 403

    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    subject = request.form.get('subject') 
    unit = request.form.get('unit')       
    
    if not subject or not unit:
        return jsonify({"error": "Missing Subject or Unit details"}), 400

    filepath = os.path.join(BASE_UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        # --- CRITICAL FIX 3: Robust PDF Processing ---
        # Handle cases where PDF is empty or scanned (image-only)
        loader = PyPDFLoader(filepath)
        documents = loader.load()
        
        if not documents:
            os.remove(filepath)
            return jsonify({"error": "File appears empty or is a scanned image (no text found)."}), 400
        # ---------------------------------------------

        # Step 1: Clean up duplicates
        existing_docs = supabase.table("documents").select("id") \
            .filter("metadata->>subject", "eq", subject) \
            .filter("metadata->>unit", "eq", unit) \
            .filter("metadata->>filename", "eq", file.filename) \
            .execute()
        
        if existing_docs.data:
            ids_to_delete = [doc['id'] for doc in existing_docs.data]
            supabase.table("documents").delete().in_("id", ids_to_delete).execute()

        # Step 2: Split & Embed
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        texts = [c.page_content for c in chunks]
        embeddings = embedding_model.embed_documents(texts)

        # Step 3: Insert with Metadata
        records = []
        user_id = session.get('user_id') 

        for i, chunk in enumerate(chunks):
            page_number = chunk.metadata.get('page', 0) + 1
            records.append({
                "user_id": user_id, 
                "content": chunk.page_content,
                "metadata": {
                    "subject": subject,
                    "unit": unit,
                    "filename": file.filename,
                    "type": "syllabus",
                    "uploaded_by": user_id,
                    "page": page_number
                },
                "embedding": embeddings[i]
            })

        supabase.table("documents").insert(records).execute()
        os.remove(filepath)
        return jsonify({"message": f"Successfully uploaded {file.filename}"})
        
    except Exception as e:
        print(f"Upload Error: {e}")
        # Ensure temp file is removed even on error
        if os.path.exists(filepath): os.remove(filepath)
        return jsonify({"error": str(e)}), 500

@app.route('/get_library', methods=['GET'])
def get_library():
    try:
        res = supabase.table("documents").select("metadata").execute()
        library = {}
        for row in res.data:
            meta = row.get('metadata', {})
            subj = meta.get('subject', 'Uncategorized')
            unit = meta.get('unit', 'General')
            fname = meta.get('filename', 'Unknown')
            
            if subj not in library: library[subj] = {}
            if unit not in library[subj]: library[subj][unit] = []
            
            if fname not in library[subj][unit]:
                library[subj][unit].append(fname)
                
        return jsonify(library)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_file', methods=['POST'])
def delete_file():
    if session.get('role') != 'admin': return jsonify({"error": "Unauthorized"}), 403
    
    data = request.json
    subject = data.get('subject')
    unit = data.get('unit')
    filename = data.get('filename')

    if not (subject and unit and filename): return jsonify({"error": "Missing parameters"}), 400

    try:
        supabase.table("documents").delete() \
            .filter("metadata->>subject", "eq", subject) \
            .filter("metadata->>unit", "eq", unit) \
            .filter("metadata->>filename", "eq", filename) \
            .execute()
        return jsonify({"message": f"Deleted {filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    user_id = session.get('user_id')
    if not user_id: return jsonify({"error": "Unauthorized"}), 401
    
    data = request.json
    user_query = data.get('query')
    subject = data.get('subject')
    unit = data.get('unit')
    history = data.get('history', [])
    
    # Validation
    if not subject: return jsonify({"error": "Please select a subject."}), 400
    if not user_query: return jsonify({"error": "Message cannot be empty."}), 400

    # 1. Embed Query
    query_vector = embedding_model.embed_query(user_query)

    # 2. Retrieve Context with Similarity Scores
    rpc_params = {
        "query_embedding": query_vector,
        "match_threshold": 0.4, # Keep this low to retrieve *something*
        "match_count": 4,
        "filter": {"subject": subject, "unit": unit}
    }
    
    try:
        response = supabase.rpc("match_documents", rpc_params).execute()
        
        # --- PHASE 2 FEATURE: CONTENT GAP DETECTION ---
        # Calculate the confidence of the best match
        top_score = response.data[0]['similarity'] if response.data else 0
        CONFIDENCE_THRESHOLD = 0.60  # Adjust based on testing (0.60 is a good starting point)

        if top_score < CONFIDENCE_THRESHOLD:
            # Log this as a "Miss" so the teacher knows
            print(f"⚠️ Low Confidence ({top_score:.2f}) for: {user_query}")
            supabase.table("missed_queries").insert({
                "query": user_query,
                "subject": subject,
                "unit": unit,
                "score": top_score
            }).execute()
        # ---------------------------------------------

        # Format context for the LLM
        context_text = ""
        for doc in response.data:
            meta = doc['metadata']
            filename = meta.get('filename', 'Unknown')
            page_num = meta.get('page', '?')
            context_text += f"[Source: {filename}, Page: {page_num}]\n{doc['content']}\n\n"
            
        # If context is empty (extremely low match), give a fallback
        if not context_text:
            context_text = "No relevant documents found. Please inform the user that this topic isn't covered in the current notes."

        # 3. Build AI Prompt
        messages = [
            {
                "role": "system", 
                "content": f"""
                You are an expert Engineering Tutor.
                
                Context provided below:
                {context_text}
                
                Instructions:
                1. Answer the user's question strictly based on the Context.
                2. If the context is empty or irrelevant, politely say you don't have information on this topic yet.
                3. If you find the answer, YOU MUST cite the source and page number at the end.
                4. Format: (Source: filename.pdf, Page X).
                """
            }
        ]
        
        for msg in history[-3:]: 
            messages.append({"role": msg['role'], "content": msg['content']})

        messages.append({"role": "user", "content": user_query})

        # 4. Generate Answer
        chat_completion = client_groq.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            temperature=0.3,
        )
        
        ai_response = chat_completion.choices[0].message.content
        
        # 5. Log Chat
        try:
            supabase.table('chats').insert({
                "user_id": user_id, "message": user_query, "role": "user", "subject": subject, "unit": unit
            }).execute()
            supabase.table('chats').insert({
                "user_id": user_id, "message": ai_response, "role": "assistant", "subject": subject, "unit": unit
            }).execute()
        except Exception as db_error:
            print(f"Logging Error: {db_error}")
        
        return jsonify({"response": ai_response})
        
    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({"error": str(e)}), 500

# =========================================================
#  UTILITIES & QUIZ
# =========================================================

@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    data = request.json
    subject = data.get('subject')
    unit = data.get('unit')
    
    if not subject or not unit:
        return jsonify({"error": "Please select a Subject and Unit first."}), 400

    try:
        # Search for broad concepts in the unit
        query_vec = embedding_model.embed_query("important definitions key concepts summary")
        rpc_params = {
            "query_embedding": query_vec,
            "match_threshold": 0.3, 
            "match_count": 6,
            "filter": {"subject": subject, "unit": unit}
        }
        
        response = supabase.rpc("match_documents", rpc_params).execute()
        context_text = "\n".join([doc['content'] for doc in response.data])
        
        if not context_text: return jsonify({"error": "Not enough content to generate a quiz."}), 400

        prompt = f"""
        Based strictly on the following text, generate 5 Multiple Choice Questions (MCQs).
        Return ONLY a raw JSON object.
        Structure: {{ "questions": [ {{ "id": 1, "question": "...", "options": ["A", "B", "C", "D"], "answer": "..." }} ] }}
        Text: {context_text}
        """

        chat_completion = client_groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        return jsonify(json.loads(chat_completion.choices[0].message.content))
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clear_data', methods=['DELETE'])
def clear_data():
    user_id = session.get('user_id')
    if not user_id: return jsonify({"error": "Unauthorized"}), 401
    
    try:
        supabase.table('chats').delete().eq('user_id', user_id).execute()
        # For documents, we filter by metadata since the column is JSON
        supabase.table('documents').delete().filter('metadata->>user_id', 'eq', user_id).execute()
        return jsonify({"message": "Conversation and storage cleared!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Change port to 5001
    app.run(debug=True, port=5001)