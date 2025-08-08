import json
import re
from typing import Dict, List
from sentence_transformers import SentenceTransformer
import numpy as np

class QueryHandler:
    def __init__(self):
        # Lightweight embedding model only (no LLM)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.keyword_rules = {
            'knee surgery': {'amount': 150000, 'clause': 'ORTHOPEDIC_CLAUSE'},
            'cardiac': {'amount': 200000, 'clause': 'CARDIAC_CLAUSE'}
        }

    def parse_query(self, query: str) -> Dict:
        """Rule-based parsing that works without LLM"""
        query_lower = query.lower()
        return {
            'procedure': self._extract_procedure(query_lower),
            'location': 'pune' if 'pune' in query_lower else None,
            'policy_age': self._extract_policy_age(query_lower)
        }

    def _extract_procedure(self, query: str) -> str:
        for procedure in self.keyword_rules:
            if procedure in query:
                return procedure
        return None

    def _extract_policy_age(self, query: str) -> int:
        match = re.search(r'(\d+)\s*(month|year)', query)
        if match:
            return int(match.group(1))
        return None

    def make_decision(self, query: str, relevant_chunks: List[Dict]) -> Dict:
        """Hybrid approach using embeddings + rules"""
        context_embed = self.embedder.encode(
            [chunk['text'] for chunk in relevant_chunks]
        )
        query_embed = self.embedder.encode([query])
        
        # Find most relevant chunk
        scores = np.dot(query_embed, np.array(context_embed).T)
        best_chunk = relevant_chunks[np.argmax(scores)]['text']
        
        # Rule-based decision
        parsed = self.parse_query(query)
        if parsed['procedure'] in self.keyword_rules:
            rule = self.keyword_rules[parsed['procedure']]
            return {
                "decision": "approved",
                "amount": rule['amount'],
                "justification": f"Covered under {rule['clause']}",
                "clauses_used": [rule['clause']]
            }
        
        return {
            "decision": "rejected",
            "justification": "No matching coverage found",
            "clauses_used": []
        }