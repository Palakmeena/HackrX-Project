import re
import json
import os
from typing import Dict, List, Optional
from models import StructuredQuery
from dataclasses import asdict

class QueryHandler:
    def __init__(self):
        self.coverage_keywords = {
            'knee': ['knee', 'orthopedic', 'joint'],
            'cardiac': ['cardiac', 'heart', 'bypass'],
            'eye': ['eye', 'cataract', 'lasik']
        }
        self.minimum_coverages = {
            'knee': 100000,
            'cardiac': 200000,
            'eye': 50000
        }

    def parse_query(self, query: str) -> StructuredQuery:
        """Enhanced query parser with medical term recognition"""
        query_lower = query.lower()
        
        return StructuredQuery(
            age=self._extract_age(query_lower),
            gender=self._extract_gender(query_lower),
            procedure=self._extract_procedure(query_lower),
            location=self._extract_location(query_lower),
            policy_duration=self._extract_policy_duration(query_lower),
            raw_query=query
        )

    def _extract_age(self, query: str) -> Optional[int]:
        """Extract age with more pattern coverage"""
        patterns = [
            r'(\d+)[ -]year[- ]old',
            r'age[ :]*(\d+)',
            r'(\d+)[mf]\b',  # 45m or 30f
            r'\b(\d{2})\b'  # standalone 2-digit number
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                age = int(match.group(1))
                return age if 18 <= age <= 100 else None
        return None

    def _extract_gender(self, query: str) -> Optional[str]:
        """Gender extraction with more variants"""
        if re.search(r'\bfemale\b|\bwoman\b|\bgirl\b|\b(\d+)f\b', query):
            return "female"
        elif re.search(r'\bmale\b|\bman\b|\bboy\b|\b(\d+)m\b', query):
            return "male"
        return None

    def _extract_procedure(self, query: str) -> Optional[str]:
        """Procedure extraction with medical term mapping"""
        # First check for exact procedure matches
        procedure_map = {
            'knee surgery': ['knee replacement', 'acl surgery', 'meniscus repair'],
            'cardiac surgery': ['heart bypass', 'angioplasty', 'stent placement'],
            'eye surgery': ['cataract', 'lasik', 'retinal detachment']
        }
        
        for procedure, terms in procedure_map.items():
            if any(term in query for term in [procedure] + terms):
                return procedure
                
        # Fallback to keyword matching
        for category, keywords in self.coverage_keywords.items():
            if any(keyword in query for keyword in keywords):
                return category
                
        return None

    def _extract_location(self, query: str) -> Optional[str]:
        """Location extraction with city normalization"""
        cities = {
            'mumbai': ['mumbai', 'bombay'],
            'delhi': ['delhi', 'new delhi'],
            'pune': ['pune', 'puna'],
            'bangalore': ['bangalore', 'bengaluru']
        }
        
        for city, aliases in cities.items():
            if any(alias in query for alias in aliases):
                return city
        return None

    def _extract_policy_duration(self, query: str) -> Optional[str]:
        """Duration extraction with month/year conversion"""
        patterns = [
            r'(\d+)[ -]month',
            r'(\d+)[ -]year',
            r'policy[ :]*(\d+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                duration = int(match.group(1))
                if 'year' in pattern:
                    return f"{duration * 12} months"
                return f"{duration} months"
        return None

    def make_decision(self, original_query: str, 
                     structured_query: StructuredQuery,
                     relevant_chunks: List[Dict]) -> Dict:
        """Enhanced decision logic with policy text analysis"""
        
        # Combine all relevant chunks into one searchable text
        policy_text = "\n".join([chunk['text'] for chunk in relevant_chunks])
        
        # Initialize response
        response = {
            "decision": "rejected",
            "amount": 0.0,
            "justification": [],
            "clauses_used": []
        }

        # 1. Check basic eligibility
        if not self._check_basic_eligibility(structured_query, policy_text, response):
            return response

        # 2. Procedure-specific checks
        if structured_query.procedure:
            self._check_procedure_coverage(structured_query.procedure, policy_text, response)

        # 3. Location check
        if structured_query.location:
            self._check_location_coverage(structured_query.location, policy_text, response)

        # 4. Finalize response
        if not response["justification"]:
            response["justification"] = "No matching coverage found in policy documents"
            
        response["justification"] = ". ".join(response["justification"])
        return response

    def _check_basic_eligibility(self, query: StructuredQuery, 
                               policy_text: str, 
                               response: Dict) -> bool:
        """Check age, policy duration etc."""
        clauses = []
        
        # Age check
        if query.age:
            age_limit_match = re.search(r'age limit.*?(\d+)[^\d]+(\d+)', policy_text, re.IGNORECASE)
            if age_limit_match:
                min_age, max_age = map(int, age_limit_match.groups())
                if not (min_age <= query.age <= max_age):
                    response["justification"].append(
                        f"Age {query.age} outside coverage range ({min_age}-{max_age})"
                    )
                    clauses.append("Age eligibility clause")
                    return False

        # Policy duration check
        if query.policy_duration:
            min_duration_match = re.search(r'waiting period.*?(\d+)\s*months', policy_text, re.IGNORECASE)
            if min_duration_match:
                min_months = int(min_duration_match.group(1))
                query_months = int(query.policy_duration.split()[0])
                if query_months < min_months:
                    response["justification"].append(
                        f"Policy duration {query_months} months less than minimum {min_months}"
                    )
                    clauses.append("Waiting period clause")
                    return False

        return True

    def _check_procedure_coverage(self, procedure: str, 
                                 policy_text: str, 
                                 response: Dict) -> None:
        """Check coverage for specific medical procedures"""
        # Look for procedure-specific coverage
        coverage_pattern = (
            r'(?:{})[^.]*?(?:covered|coverage|limit)[^.]*?(\d{{1,3}}(?:,\d{{3}})*)'
            .format('|'.join(self.coverage_keywords.get(procedure, [procedure])))
        )
        
        matches = re.finditer(coverage_pattern, policy_text, re.IGNORECASE)
        for match in matches:
            amount = int(match.group(1).replace(",", ""))
            if amount > response["amount"]:
                response.update({
                    "decision": "approved",
                    "amount": float(amount),
                    "clauses_used": [f"Coverage for {procedure}: Rs. {amount}"]
                })
                response["justification"].append(
                    f"{procedure.capitalize()} covered up to Rs. {amount}"
                )

    def _check_location_coverage(self, location: str, 
                               policy_text: str, 
                               response: Dict) -> None:
        """Check if location is in network"""
        network_pattern = r'network (?:hospitals|providers).*?{}'.format(location)
        if not re.search(network_pattern, policy_text, re.IGNORECASE):
            response["justification"].append(
                f"No network coverage in {location.capitalize()}"
            )
            if response["decision"] == "approved":
                response.update({
                    "decision": "rejected",
                    "amount": 0.0
                })