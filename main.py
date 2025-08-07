from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from document_processor import DocumentProcessor
from query_handler import QueryHandler
from vector_store import VectorStore
import os

app = FastAPI()

# Initialize with memory limits
processor = DocumentProcessor()
vector_db = VectorStore()
query_handler = QueryHandler()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Memory-constrained upload endpoint"""
    try:
        # Save temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # Process in chunks
        chunks = processor.process_document(temp_path)
        if not chunks:
            return JSONResponse({"error": "Processing failed"}, status_code=400)
            
        vector_db.add_documents(chunks)
        return {"message": f"Processed {len(chunks)} chunks"}
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/query")
async def handle_query(query: str):
    """Optimized query endpoint"""
    try:
        relevant_chunks = vector_db.search(query)
        if not relevant_chunks:
            return {"decision": "rejected", "justification": "No matching clauses"}
            
        return query_handler.make_decision(query, relevant_chunks)
    except Exception as e:
        return {"error": str(e)}