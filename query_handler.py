import json
import re
from typing import Dict, List
from llama_cpp import Llama
import os

class QueryHandler:
    def __init__(self):
        # Load quantized Mistral-7B (3.5GB)
        self.llm = self._init_llm()
        
    def _init_llm(self):
        """Initialize with memory optimizations"""
        return Llama(
            model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            n_ctx=1024,  # Reduced context window
            n_threads=4,
            n_batch=512,
            verbose=False
        )

    def parse_query(self, query: str) -> Dict:
        """Lightweight query parsing"""
        return {
            "raw_query": query,
            "procedure": self._extract_keyword(query, ['knee', 'surgery', 'cardiac']),
            "location": self._extract_keyword(query, ['pune', 'mumbai', 'delhi'])
        }

    def _extract_keyword(self, text: str, keywords: List[str]) -> str:
        """Efficient keyword extraction"""
        text_lower = text.lower()
        for kw in keywords:
            if kw in text_lower:
                return kw
        return ""

    def make_decision(self, query: str, relevant_chunks: List[Dict]) -> Dict:
        """LLM-powered decision with memory safety"""
        try:
            context = "\n".join([c['text'] for c in relevant_chunks[:3]])  # Use top 3 chunks
            
            prompt = f"""Analyze this insurance context and respond in JSON:
            Context: {context[:2000]}  # Truncate for memory
            Query: {query}
            
            Output format:
            {{
                "decision": "approved/rejected",
                "amount": number,
                "justification": "string",
                "clauses_used": ["list"]
            }}"""
            
            # Memory-constrained generation
            response = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.1
            )
            
            return json.loads(response['choices'][0]['message']['content'])
            
        except Exception as e:
            return {
                "decision": "error",
                "justification": f"System error: {str(e)}"
            }