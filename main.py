from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import uvicorn
from models import QueryRequest, QueryResponse
from document_processor import DocumentProcessor
from query_handler import QueryHandler
from vector_store import VectorStore

# Initialize FastAPI app
app = FastAPI(title="HackRx LLM Document Processing API", version="1.0.0")

# Initialize components
doc_processor = DocumentProcessor()
query_handler = QueryHandler()
vector_store = VectorStore()

@app.get("/")
async def root():
    return {"message": "HackRx LLM Document Processing API is running!"}

@app.post("/upload-documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents (PDF, Word, etc.)"""
    try:
        processed_count = 0
        for file in files:
            # Save uploaded file temporarily
            file_path = f"temp_{file.filename}"
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Process document
            chunks = doc_processor.process_document(file_path)
            
            # Store in vector database
            vector_store.add_documents(chunks, file.filename)
            processed_count += 1
        
        return {"message": f"Successfully processed {processed_count} documents"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")

@app.post("/process-query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process natural language query and return structured response"""
    try:
        # Parse the query
        structured_query = query_handler.parse_query(request.query)
        
        # Search for relevant documents
        relevant_chunks = vector_store.search(request.query, top_k=5)
        
        # Make decision using LLM
        decision = query_handler.make_decision(request.query, structured_query, relevant_chunks)
        
        return QueryResponse(**decision)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)