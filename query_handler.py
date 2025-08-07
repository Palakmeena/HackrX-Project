import re
import json
import os
import requests
from dotenv import load_dotenv
from typing import Dict, List
from models import StructuredQuery

class QueryHandler:
    def __init__(self):
        # Load Hugging Face API token from .env file
        load_dotenv()
        try:
            self.hf_token = os.getenv("HF_API_KEY")
            if not self.hf_token:
                raise Exception("No Hugging Face token found")
        except:
            self.hf_token = None
            print("Using rule-based approach for query processing (no Hugging Face token)")

    def query_llm_hf(self, prompt: str) -> str:
        """Call Hugging Face Inference API for response generation"""
        if not self.hf_token:
            return "LLM not available"

        API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {"inputs": prompt}

        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and "generated_text" in data[0]:
                return data[0]["generated_text"]
            return str(data)
        else:
            return f"Error: {response.status_code} - {response.text}"

    def parse_query(self, query: str) -> StructuredQuery:
        """Parse natural language query into structured format"""
        parsed = {
            "age": self._extract_age(query),
            "gender": self._extract_gender(query),
            "procedure": self._extract_procedure(query),
            "location": self._extract_location(query),
            "policy_duration": self._extract_policy_duration(query),
            "raw_query": query
        }
        return StructuredQuery(**parsed)

    def _extract_age(self, query: str) -> int:
        patterns = [
            r'(\d+)M',
            r'(\d+)F',
            r'(\d+)-year-old',
            r'(\d+)\s*years?\s*old',
            r'age\s*(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def _extract_gender(self, query: str) -> str:
        if re.search(r'\b(\d+)M\b', query):
            return "male"
        elif re.search(r'\b(\d+)F\b', query):
            return "female"
        elif re.search(r'\bmale\b', query, re.IGNORECASE):
            return "male"
        elif re.search(r'\bfemale\b', query, re.IGNORECASE):
            return "female"
        return None

    def _extract_procedure(self, query: str) -> str:
        procedures = [
            "knee surgery", "cardiac", "heart surgery", "hip replacement",
            "cataract", "dental", "surgery", "operation", "procedure"
        ]
        query_lower = query.lower()
        for procedure in procedures:
            if procedure in query_lower:
                return procedure
        return None

    def _extract_location(self, query: str) -> str:
        cities = [
            "mumbai", "delhi", "bangalore", "hyderabad", "pune", "chennai",
            "kolkata", "ahmedabad", "jaipur", "lucknow", "kanpur", "nagpur"
        ]
        query_lower = query.lower()
        for city in cities:
            if city in query_lower:
                return city.title()
        return None

    def _extract_policy_duration(self, query: str) -> str:
        patterns = [
            r'(\d+)-month policy',
            r'(\d+)\s*month\s*policy',
            r'(\d+)-year policy',
            r'(\d+)\s*year\s*policy',
        ]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    def make_decision(self, original_query: str, structured_query: StructuredQuery, 
                      relevant_chunks: List[Dict]) -> Dict:
        decision_result = {
            "decision": "rejected",
            "amount": 0.0,
            "justification": "",
            "clauses_used": []
        }

        if not relevant_chunks:
            decision_result["justification"] = "No relevant policy information found for this query."
            return decision_result

        relevant_text = " ".join([chunk["text"] for chunk in relevant_chunks])

        if structured_query.age and (structured_query.age < 18 or structured_query.age > 65):
            decision_result["justification"] = f"Age {structured_query.age} is outside policy coverage (18-65 years)."
            decision_result["clauses_used"] = ["Age limit: 18-65 years"]
            return decision_result

        if structured_query.policy_duration:
            if "month" in structured_query.policy_duration.lower():
                months = int(re.search(r'(\d+)', structured_query.policy_duration).group(1))
                if months < 6:
                    decision_result["justification"] = f"Policy duration {months} months is below minimum requirement of 6 months."
                    decision_result["clauses_used"] = ["Minimum policy duration: 6 months for coverage"]
                    return decision_result

        if structured_query.procedure:
            procedure_lower = structured_query.procedure.lower()

            if "knee" in procedure_lower and "surgery" in procedure_lower:
                if "1,00,000" in relevant_text or "100000" in relevant_text:
                    decision_result["decision"] = "approved"
                    decision_result["amount"] = 100000.0
                    decision_result["justification"] = "Knee surgery is covered under the policy up to Rs. 1,00,000."
                    decision_result["clauses_used"] = ["Knee surgeries are covered up to Rs. 1,00,000"]

            elif "cardiac" in procedure_lower:
                if "2,00,000" in relevant_text or "200000" in relevant_text:
                    decision_result["decision"] = "approved"
                    decision_result["amount"] = 200000.0
                    decision_result["justification"] = "Cardiac procedure is covered under the policy up to Rs. 2,00,000."
                    decision_result["clauses_used"] = ["Cardiac procedures covered up to Rs. 2,00,000"]
            else:
                decision_result["justification"] = f"Procedure '{structured_query.procedure}' is not explicitly covered in the policy."

        if structured_query.location:
            covered_areas = ["mumbai", "delhi", "pune", "bangalore"]
            if structured_query.location.lower() not in covered_areas:
                if decision_result["decision"] == "approved":
                    decision_result["decision"] = "rejected"
                    decision_result["amount"] = 0.0
                    decision_result["justification"] = f"Location '{structured_query.location}' is not in coverage area."
                    decision_result["clauses_used"] = ["Coverage areas: Mumbai, Delhi, Pune, Bangalore"]

        return decision_result
