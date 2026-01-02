import os
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from groq import Groq
from supabase import create_client, Client

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Configuration ---
BASE_UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(BASE_UPLOAD_FOLDER, exist_ok=True)

# --- Init Clients ---
# 1. Groq
client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# 2. Supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# 3. Embeddings (Lightweight for Free Tier)
embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# --- Routes ---
# --- Add this to your app.py (anywhere before if __name__ == '__main__':) ---

@app.route('/get_user_info', methods=['GET'])
def get_user_info():
    if 'user_id' not in session: return jsonify({"error": "Unauthorized"}), 401
    
    # In a real app, you might query a 'profiles' table here.
    # For now, we return the session data we already have.
    return jsonify({
        "email": session.get('email'),
        "user_id": session.get('user_id'),
        "initial": session.get('email')[0].upper() if session.get('email') else "?"
    })

@app.route('/')
def index():
    if 'user_id' not in session: return render_template('login.html')
    return render_template('index.html')

@app.route('/login_page')
def login_page(): return render_template('login.html')

# Add this near the top of app.py with your other configs
ADMIN_ACCESS_TOKEN = "TEACHER_SECURE_2026"  # <--- Change this to whatever secret password you want

@app.route('/auth/signup', methods=['POST'])
def signup():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'student')
    admin_code = data.get('admin_code') # New field

    # --- SECURITY CHECK ---
    if role == 'admin':
        if admin_code != ADMIN_ACCESS_TOKEN:
            return jsonify({"error": "Invalid Admin Access Code! You are not authorized."}), 403
    # ----------------------

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
        
        # 2. CRITICAL FIX: Fetch Role from 'profiles' table
        profile_res = supabase.table('profiles').select('role').eq('id', user.id).execute()
        user_role = profile_res.data[0]['role'] if profile_res.data else 'student'

        # 3. Save Role in Session (So the upload function knows who you are)
        session['user_id'] = user.id
        session['email'] = user.email
        session['role'] = user_role 

        # 4. Send Role to Frontend (So the UI shows the Teacher Dashboard)
        return jsonify({"message": "Login successful", "redirect": "/", "role": user_role})
    except Exception as e: 
        print(f"Login Error: {e}")
        return jsonify({"error": "Invalid credentials"}), 401
@app.route('/auth/logout')
def logout():
    session.clear()
    return redirect('/login_page')

@app.route('/get_history', methods=['GET'])
def get_history():
    user_id = session.get('user_id')
    if not user_id: return jsonify({"error": "Unauthorized"}), 401
    try:
        response = supabase.table('chats').select("*").eq('user_id', user_id).order('created_at', desc=False).execute()
        return jsonify({"history": response.data})
    except Exception as e: return jsonify({"error": str(e)}), 500

# --- Core RAG Features (Direct Supabase Implementation) ---

@app.route('/upload', methods=['POST'])
def upload_file():
    # 1. Check Admin Permission
    if session.get('role') != 'admin':
        return jsonify({"error": "Unauthorized: Only Admins can upload syllabus."}), 403

    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    
    # 2. Get Academic Metadata from Form
    subject = request.form.get('subject') 
    unit = request.form.get('unit')       
    
    if not subject or not unit:
        return jsonify({"error": "Missing Subject or Unit details"}), 400

    # 3. Save Temp File
    filepath = os.path.join(BASE_UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        loader = PyPDFLoader(filepath)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        texts = [c.page_content for c in chunks]
        embeddings = embedding_model.embed_documents(texts)

        # 4. Insert with ACADEMIC METADATA AND USER_ID
        records = []
        user_id = session.get('user_id') # <--- Get ID from session

        for i, chunk in enumerate(chunks):
            records.append({
                "user_id": user_id,  # <--- CRITICAL FIX: Satisfy the database constraint
                "content": chunk.page_content,
                "metadata": {
                    "subject": subject,
                    "unit": unit,
                    "filename": file.filename,
                    "type": "syllabus",
                    "uploaded_by": user_id 
                },
                "embedding": embeddings[i]
            })

        supabase.table("documents").insert(records).execute()
        os.remove(filepath)
        return jsonify({"message": f"Uploaded to Subject: {subject}"})
        
    except Exception as e:
        print(f"Upload Error: {e}")
        return jsonify({"error": str(e)}), 500

# --- NEW ROUTE: Fetch File Library ---
@app.route('/get_library', methods=['GET'])
def get_library():
    # Fetch all unique file metadata (Subject, Unit, Filename)
    try:
        # We perform a distinct query on metadata columns
        # Note: Supabase Python client doesn't support 'distinct' easily on JSON cols
        # So we fetch metadata and process in Python (Efficient enough for <10k files)
        res = supabase.table("documents").select("metadata").execute()
        
        library = {}
        for row in res.data:
            meta = row.get('metadata', {})
            subj = meta.get('subject', 'Uncategorized')
            unit = meta.get('unit', 'General')
            fname = meta.get('filename', 'Unknown')
            
            if subj not in library: library[subj] = {}
            if unit not in library[subj]: library[subj][unit] = []
            
            # Avoid duplicates in the list
            if fname not in library[subj][unit]:
                library[subj][unit].append(fname)
                
        return jsonify(library)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- UPDATED CHAT ROUTE (With Unit Filtering) ---
@app.route('/chat', methods=['POST'])
def chat():
    user_id = session.get('user_id')
    if not user_id: return jsonify({"error": "Unauthorized"}), 401
    
    data = request.json
    user_query = data.get('query')
    subject = data.get('subject')
    unit = data.get('unit') # <--- NEW: Filter by Unit

    if not subject: return jsonify({"error": "Please select a subject."}), 400

    # 1. Embed Query
    query_vector = embedding_model.embed_query(user_query)

    # 2. Build Filter
    # We filter by Subject AND Unit (if provided)
    meta_filter = {"subject": subject}
    if unit:
        meta_filter["unit"] = unit

    rpc_params = {
        "query_embedding": query_vector,
        "match_threshold": 0.4, 
        "match_count": 4,
        "filter": meta_filter # <--- Stricter Filter
    }
    
    try:
        response = supabase.rpc("match_documents", rpc_params).execute()
        docs = response.data
        
        # Build Context
        context_text = ""
        for doc in docs:
            meta = doc['metadata']
            context_text += f"[Source: {meta['filename']}, {meta['unit']}]\n{doc['content']}\n\n"
            
        if not context_text:
            return jsonify({"response": f"I couldn't find information in {subject} ({unit})."})

        # 3. System Prompt
        system_prompt = f"""
        You are an expert Engineering Tutor.
        Context:
        {context_text}
        
        Question: {user_query}
        
        Answer strictly based on the Context. Cite the [Source] at the end.
        """

        chat_completion = client_groq.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3,
        )
        
        return jsonify({"response": chat_completion.choices[0].message.content})
        
    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_subjects', methods=['GET'])
def get_subjects():
    try:
        # Fetch distinct subjects from the documents table metadata
        # Note: Supabase doesn't support 'distinct' on JSON keys easily via API
        # So we fetch all metadata (lightweight) and filter in Python
        # For production, you'd make a separate 'subjects' table.
        
        res = supabase.table("documents").select("metadata").execute()
        subjects = set()
        for row in res.data:
            if 'subject' in row['metadata']:
                subjects.add(row['metadata']['subject'])
        
        return jsonify({"subjects": list(subjects)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500    
    
# --- Data Management Module ---

@app.route('/clear_data', methods=['DELETE'])
def clear_data():
    user_id = session.get('user_id')
    if not user_id: return jsonify({"error": "Unauthorized"}), 401
    
    try:
        # 1. Delete Chat History
        supabase.table('chats').delete().eq('user_id', user_id).execute()
        
        # 2. Delete Uploaded Documents (Vectors)
        # Note: We filter inside the JSON metadata to find the user's files
        supabase.table('documents').delete().filter('metadata->>user_id', 'eq', user_id).execute()
        
        return jsonify({"message": "Conversation and storage cleared successfully!"})
    except Exception as e:
        print(f"Clear Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)