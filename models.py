from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    decision: str  # "approved" or "rejected"
    amount: float
    justification: str
    clauses_used: List[str]

class StructuredQuery(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    procedure: Optional[str] = None
    location: Optional[str] = None
    policy_duration: Optional[str] = None
    raw_query: str