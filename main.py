from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from document_processor import DocumentProcessor
from query_handler import QueryHandler
from vector_store import VectorStore
import os

app = FastAPI()

processor = DocumentProcessor()
vector_db = VectorStore()
query_handler = QueryHandler()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save to temp file
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        chunks = processor.process_document(temp_path)
        if not chunks:
            raise HTTPException(400, "Failed to process document")
            
        vector_db.add_documents(chunks)
        return {"message": f"Processed {len(chunks)} chunks"}
        
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/query")
async def handle_query(query: str):
    try:
        chunks = vector_db.search(query)
        if not chunks:
            return {"decision": "rejected", "justification": "No matching documents"}
            
        return query_handler.make_decision(query, chunks)
    except Exception as e:
        raise HTTPException(500, str(e))