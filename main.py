import os
import uuid
from typing import List
from fastapi import FastAPI, Request, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel

from dotenv import load_dotenv

load_dotenv()

import logging
import sys

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

use_ollama_embed = os.getenv("USE_OLLAMA_EMBED", "false").lower() == "true"

app = FastAPI()
retriever = None

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOADS_DIR = "data/uploads"
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

INDEX_DIR = "data/chroma_index"
if not os.path.exists(INDEX_DIR):
    os.makedirs(INDEX_DIR)

MARKER_FILE = os.path.join(INDEX_DIR, "indexing_complete.marker")

tasks = {}

class ChatRequest(BaseModel):
    query: str


def parse_pdf(pdf_path: str) -> List:
    logger.info(f"Parsing PDF file - {pdf_path}")
    
    elements = partition_pdf(
        filename=pdf_path,
        strategy="fast",
    )
    
    logger.info(f"Extracted {len(elements)} elements")
    return elements

def create_chunks_from_element(elements) -> List:
    logger.info("Creating chunks...")
    
    chunks = chunk_by_title(
        elements, 
        max_characters=2000, 
        new_after_n_chars=1600, 
        combine_text_under_n_chars=400 
    )
    
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

def create_langchain_documents(chunks, file_path) -> List[Document]:
    logger.info("Converting chunks to LangChain Document format...")
    
    documents = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "chunk_id": i,
            "page_number": chunk.metadata.page_number,
            "title": chunk.metadata.title if hasattr(chunk.metadata, 'title') else None,
            "filename": os.path.basename(file_path),
        }
        doc = Document(page_content=chunk.text, metadata=metadata)
        documents.append(doc)
    
    logger.info(f"Converted {len(documents)} chunks to Document format")
    return documents


def create_vector_store(documents):
    logger.info("Create embeddings and storing in DB...")

    collection_name = "the_collection"
    if use_ollama_embed:
        embedding_model = OllamaEmbeddings(model="mxbai-embed-large:latest")
        logger.info("Using Ollama Embeddings model-mxbai-embed-large:latest")
    else:
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        logger.info("Using OpenAI Embeddings model-text-embedding-3-large")

    vectorstore = Chroma(
        collection_name=collection_name, 
        persist_directory=INDEX_DIR, 
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    vectorstore.reset_collection()

    if os.path.exists(MARKER_FILE):
        os.remove(MARKER_FILE)

    vectorstore.add_documents(documents)
    
    with open(MARKER_FILE, "w") as f:
        f.write("done")

    logger.info(f"Vector index create in {INDEX_DIR}")
    return vectorstore


def ingest_data(pdf_path: str) -> None:
    global retriever
    logger.info("Start ingesting data...")
    
    elements = parse_pdf(pdf_path)
    chunks = create_chunks_from_element(elements)
    documents = create_langchain_documents(chunks, pdf_path)
    db = create_vector_store(documents)
    retriever = db.as_retriever(search_kwargs={"k": 2})  # Retrieve top 2 chunks for context
    
    logger.info("ingest completed successfully!")


def call_llm(chunks, query) -> str:
    logger.info("Calling LLM for answer generation...")
    try:
        llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
        docs_content = "\n\n".join(doc.page_content for doc in chunks)

        template = ChatPromptTemplate.from_messages([
            ("system", 
            "You are a helpful assistant answering user questions based ONLY on the provided context.\n\n"
            "Instructions:\n"
            "- Use ONLY the information in the Context section to answer.\n"
            "- Do NOT use your own knowledge or make assumptions.\n"
            "- If the answer is not found in the context or not clear, respond exactly with: \"I don't know\".\n"
            "- Be concise, factual, and do not include unrelated information.\n"
            "- Do not hallucinate details."),
            ("human", 
            "Context:\n{context}\n\nQuestion:\n{question}")
        ])
        
        prompt = template.invoke({"question": query, "context": docs_content})
        response = llm.invoke(prompt)
        
        return response.content
        
    except Exception as e:
        logger.error(f"Error calling LLM: {e}", exc_info=True)
        return "Could not generate an answer at this time."
    
def process_pdf(file_path: str, task_id: str) -> None:
    logger.debug(f"Starting process_pdf for task {task_id} and file {file_path}")
    """Extracts text and images from a PDF, generates summaries, and stores them in a vector store."""
    try:
        logger.info(f"Processing {file_path}")
        ingest_data(file_path)
        
        tasks[task_id] = "complete"
        logger.info(f"Task {task_id} complete")

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}", exc_info=True)
        tasks[task_id] = "error"


def process_chat_query(query: str) -> str:
    global retriever
    if retriever is None:
        if os.path.exists(MARKER_FILE):
            logger.info("Loading existing vector store...")

            if use_ollama_embed:
                embedding_model = OllamaEmbeddings(model="mxbai-embed-large:latest")
                logger.info("Using Ollama Embeddings model-mxbai-embed-large:latest")
            else:
                embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
                logger.info("Using OpenAI Embeddings model-text-embedding-3-large")

            vectorstore = Chroma(persist_directory=INDEX_DIR, embedding_function=embedding_model, collection_name="the_collection")
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
            logger.info("Retriever created from existing vector store.")
        else:
            raise ValueError("Please upload and process a PDF document first.")
    
    chunks = retriever.invoke(query)
    final_answer = call_llm(chunks, query)
    return final_answer


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    logger.debug("Root endpoint called.")
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    logger.debug("Upload page endpoint called.")
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(None)):
    logger.debug("Upload file endpoint called.")
    if not file or file.content_type != "application/pdf":
        logger.warning(f"Invalid file upload attempt. Content-Type: {file.content_type if file else 'No file'}")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    task_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOADS_DIR, file.filename)
    logger.debug(f"Generated task ID: {task_id} for file: {file.filename}")
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    logger.info(f"Uploaded file saved to {file_path}")
    
    tasks[task_id] = "processing"
    background_tasks.add_task(process_pdf, file_path, task_id)
    logger.debug(f"Task {task_id} added to background tasks.")
    
    return JSONResponse(content={"task_id": task_id})


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    logger.debug(f"Status endpoint called for task ID: {task_id}")
    status = tasks.get(task_id, "not_found")
    logger.debug(f"Task {task_id} status: {status}")
    return JSONResponse(content={"status": status})

@app.get("/index-status")
async def index_status():
    logger.debug("Index status endpoint called.")
    return JSONResponse(content={"indexed": os.path.exists(MARKER_FILE)})


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    logger.debug("Chat page endpoint called.")
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat")
async def chat(request: ChatRequest):
    logger.debug(f"Chat endpoint called with query: {request.query}")
    try:
        answer = process_chat_query(request.query)
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        logger.error(f"Error processing chat query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing your query.")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server.")
    uvicorn.run(app, host="0.0.0.0", port=8000)