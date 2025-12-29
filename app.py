import os
import shutil
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

@app.route('/auth/signup', methods=['POST'])
def signup():
    data = request.json
    try:
        response = supabase.auth.sign_up({"email": data.get('email'), "password": data.get('password')})
        if response.user: return jsonify({"message": "Signup successful! Log in."})
        return jsonify({"error": "Signup failed"}), 400
    except Exception as e: return jsonify({"error": str(e)}), 400

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.json
    try:
        response = supabase.auth.sign_in_with_password({"email": data.get('email'), "password": data.get('password')})
        session['user_id'] = response.user.id
        session['email'] = response.user.email
        return jsonify({"message": "Login successful", "redirect": "/"})
    except Exception as e: return jsonify({"error": "Invalid credentials"}), 401

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
    user_id = session.get('user_id')
    if not user_id: return jsonify({"error": "Unauthorized"}), 401

    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No file selected"}), 400

    filepath = os.path.join(BASE_UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        # 1. Load & Split
        loader = PyPDFLoader(filepath)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # 2. Generate Embeddings (Batch Process)
        texts = [c.page_content for c in chunks]
        embeddings = embedding_model.embed_documents(texts)

        # 3. Prepare Data for Insert
        # 3. Prepare Data for Insert
        records = []
        for i, chunk in enumerate(chunks):
            records.append({
                "user_id": user_id,  # <--- ADD THIS LINE!
                "content": chunk.page_content,
                "metadata": {"user_id": user_id, "filename": file.filename},
                "embedding": embeddings[i]
            })

        # 4. Insert into Supabase (Directly)
        supabase.table("documents").insert(records).execute()
        
        os.remove(filepath)
        return jsonify({"message": "File processed and stored in Cloud DB!"})
    except Exception as e:
        print(f"Upload Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    user_id = session.get('user_id')
    if not user_id: return jsonify({"error": "Unauthorized"}), 401
    
    user_query = request.json.get('query')

    # 1. Generate Query Embedding
    query_vector = embedding_model.embed_query(user_query)

    # 2. Call RPC Function (Direct Search)
    # This bypasses the buggy LangChain vector store
    rpc_params = {
        "query_embedding": query_vector,
        "match_threshold": -1.0, # Adjust this if results are too loose/strict
        "match_count": 2,
        "filter": {"user_id": user_id}
    }
    
    try:
        response = supabase.rpc("match_documents", rpc_params).execute()
        print(f"DEBUG SEARCH: Found {len(response.data)} matches.")
        if len(response.data) == 0:
            print("DEBUG: Checking User ID:", user_id)
        docs = response.data # List of matching records
        
        context_text = "\n\n".join([doc['content'] for doc in docs])
        if not context_text:
            context_text = "No relevant documents found in your private storage."

        # 3. Retrieve History
        history_response = supabase.table('chats').select("*").eq('user_id', user_id).order('created_at', desc=True).limit(2).execute()
        past_chats = history_response.data[::-1]
        
        history_text = ""
        for chat in past_chats:
            history_text += f"{chat['role'].title()}: {chat['message']}\n"

        # 4. Generate Response
        system_prompt = f"""
        You are a helpful assistant. Use the Context and History below.
        
        Context:
        {context_text}
        
        Chat History:
        {history_text}
        """

        chat_completion = client_groq.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.5,
        )
        
        bot_response = chat_completion.choices[0].message.content
        
        # 5. Save Chat
        supabase.table('chats').insert({"user_id": user_id, "message": user_query, "role": "user"}).execute()
        supabase.table('chats').insert({"user_id": user_id, "message": bot_response, "role": "bot"}).execute()
        
        return jsonify({"response": bot_response})
        
    except Exception as e:
        print(f"Chat Error: {e}")
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