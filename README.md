
# HackRx 6.0 - LLM Document Processing System

**HackRx 6.0 Submission** - AI-powered document query and decision system

##  Problem Statement
Build an LLM-powered intelligent query-retrieval system for processing insurance policy documents and making automated decisions.

##  Features
- Process PDF, DOCX, and TXT documents
- Semantic search using sentence transformers
- Intelligent decision making with LLMs
- Structured JSON responses
- RESTful API endpoints

##  API Endpoints

### POST /process-query
Send a query to get insurance decision:

**Request:**
```json
{
    "query": "46M, knee surgery, Pune, 3-month policy"
}

**Response:**

{
    "decision": "approved",
    "amount": 100000.0,
    "justification": "Knee surgery is covered under the policy up to Rs. 1,00,000",
    "clauses_used": ["Knee surgeries are covered up to Rs. 1,00,000"]
}


