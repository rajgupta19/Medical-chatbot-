from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from src.prompt import *
import google.generativeai as genai
import os


app = Flask(__name__)

# load dotenv()
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

def _resolve_gemini_model_name() -> str:
    """
    Pick a chat-capable model that actually exists for this API key.
    We inspect `genai.list_models()` and choose a sensible default.
    """
    # Order by: cheap flash models first, then more capable ones.
    preferred = [
        "gemini-flash-latest",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash-lite",
        "gemini-2.5-pro",
        "gemini-pro-latest",
    ]

    try:
        available = set()
        for m in genai.list_models():
            name = getattr(m, "name", "") or ""
            if not name:
                continue
            # Names are usually like "models/gemini-2.5-flash"
            short = name.split("/", 1)[1] if name.startswith("models/") else name
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                available.add(short)

        for cand in preferred:
            if cand in available:
                return cand
        # Fallback to any model that supports generateContent
        if available:
            return sorted(available)[0]
    except Exception:
        pass

    # Absolute last resort; may 404 but we tried.
    return "gemini-flash-latest"

GEMINI_MODEL_NAME = _resolve_gemini_model_name()
print(f"Using Gemini model: {GEMINI_MODEL_NAME}")


embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})



@app.route("/")
def index():
    return render_template('chat.html')

#when user presses the send button 

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(msg)

    if not GOOGLE_API_KEY:
        return "Missing GOOGLE_API_KEY in .env. Add GOOGLE_API_KEY=... and restart the app."

    try:
        # Retrieve relevant context from Pinecone
        docs = retriever.invoke(msg)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Build prompt using your existing system prompt template
        full_prompt = system_prompt.format(context=context) + f"\n\nQuestion: {msg}"

        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        try:
            response = model.generate_content(full_prompt, request_options={"timeout": 60})
        except (TypeError, ValueError):
            response = model.generate_content(full_prompt)

        answer = getattr(response, "text", None) or str(response)
        print("Response : ", answer)
        return answer
    except Exception as e:
        err = f"[ERROR] {type(e).__name__}: {e} (model={GEMINI_MODEL_NAME})"
        print(err)
        return err



if __name__ == '__main__':
    # Disable auto-reloader: on Windows it can constantly restart when site-packages files change.
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)
