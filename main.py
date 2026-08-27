import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ==========================================
# LOAD API KEY FROM .env
# ==========================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY nahi mili. Check karein ke .env file correct path par hai.")

# ==========================================
# FASTAPI APP CONFIG
# ==========================================
app = FastAPI(title="DocuLearn AI Backend")

# Real Open-Source Sentence Embeddings Model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
DB_DIR = "./chroma_db"
chat_history = []

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "DocuLearn AI Backend Active!"}

# ==========================================
# UPLOAD PDF & INDEX (WITH PAGE METADATA)
# ==========================================
@app.post("/upload-and-index/")
async def upload_and_index(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Sirf PDF files allowed hain!")

    try:
        # Nayi PDF upload hone par purana DB clear karna
        if os.path.exists(DB_DIR):
            try:
                shutil.rmtree(DB_DIR)
            except Exception:
                pass

        reader = PdfReader(file.file)
        texts = []
        metadatas = []

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        for page_idx, page in enumerate(reader.pages):
            extracted = page.extract_text()
            if extracted:
                chunks = text_splitter.split_text(extracted)
                for chunk in chunks:
                    texts.append(chunk)
                    metadatas.append({"page": page_idx + 1})

        if not texts:
            raise HTTPException(status_code=400, detail="PDF mein text nahi mila.")

        Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=embeddings,
            persist_directory=DB_DIR
        )

        chat_history.clear()

        return {
            "filename": file.filename,
            "total_pages": len(reader.pages),
            "total_chunks": len(texts),
            "status": "Success! Fresh PDF store ho gayi hai."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")

# ==========================================
# ASK QUESTION (WITH ACTIVE GROQ MODEL & MEMORY)
# ==========================================
@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    if not os.path.exists(DB_DIR):
        raise HTTPException(status_code=400, detail="Pehle koi PDF upload karein!")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question empty nahi ho sakta!")

    try:
        # Updated Active Groq Model
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-120b",
    temperature=0
        )

        vector_db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )

        docs = vector_db.similarity_search(request.question, k=3)

        if not docs:
            return {
                "question": request.question,
                "answer": "PDF mein information nahi mili.",
                "pages": []
            }

        context = "\n\n".join([doc.page_content for doc in docs])
        pages_found = sorted(list(set([doc.metadata.get("page", 1) for doc in docs if doc.metadata])))

        history_text = ""
        for chat in chat_history[-6:]:
            history_text += f"\nUser: {chat['question']}\nAssistant: {chat['answer']}\n"

        prompt = f"""
You are DocuLearn AI, a helpful document-based AI assistant.
Answer the user question strictly using ONLY the provided PDF context.

PDF CONTEXT:
{context}

PREVIOUS CHAT HISTORY:
{history_text}

CURRENT USER QUESTION:
{request.question}

Answer:
"""

        response = llm.invoke(prompt)
        answer = response.content

        chat_history.append({"question": request.question, "answer": answer})
        if len(chat_history) > 10:
            chat_history.pop(0)

        return {
            "question": request.question,
            "answer": answer,
            "pages": pages_found
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Exact Error Details: {str(e)}")

# ==========================================
# CLEAR CHAT HISTORY
# ==========================================
@app.delete("/clear-chat/")
def clear_chat():
    chat_history.clear()
    return {"message": "Chat history successfully cleared!"}