import logging
from typing import Dict, List, Optional
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class QualityEvaluator:
    def __init__(self):
        # Pre-compile regex patterns for efficiency
        self.sentence_pattern = re.compile(r'[.!?]+')
        self.word_pattern = re.compile(r'\b[a-zA-Z]{3,}\b')
        
        # Enhanced stop words list
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'can', 'may', 'might', 'must'
        }
        
    def evaluate_response(self, question: str, response: str, sources: List[Dict]) -> Dict[str, float]:
        """Comprehensive quality evaluation of a RAG response"""
        try:
            logger.debug(f"Evaluating response quality for question: {question[:100]}...")
            
            # Input validation
            if not question or not response:
                logger.warning("Empty question or response provided")
                return self._get_default_scores()
            
            scores = {}
            
            # Calculate individual metrics (now sync functions)
            scores["relevance"] = self._calculate_relevance_score(question, response)
            scores["accuracy"] = self._calculate_accuracy_score(response, sources)
            scores["completeness"] = self._calculate_completeness_score(question, response)
            scores["coherence"] = self._calculate_coherence_score(response)
            scores["citation"] = self._calculate_citation_score(response, sources)
            
            # Calculate overall score
            scores["overall"] = self._calculate_overall_score(scores)
            
            logger.debug(f"Quality scores: {scores}")
            return scores
            
        except Exception as e:
            logger.exception("Quality evaluation failed")
            return self._get_default_scores()
    
    def _get_default_scores(self) -> Dict[str, float]:
        """Return default scores when evaluation fails"""
        return {
            "relevance": 0.5,
            "accuracy": 0.5,
            "completeness": 0.5,
            "coherence": 0.5,
            "citation": 0.5,
            "overall": 0.5
        }
    
    def _calculate_relevance_score(self, question: str, response: str) -> float:
        """Calculate semantic relevance between question and response"""
        try:
            # Extract key terms
            question_terms = set(self._extract_key_terms(question))
            response_terms = set(self._extract_key_terms(response))
            
            if not question_terms:
                return 0.0
            
            # Term overlap score
            common_terms = question_terms & response_terms
            term_overlap = len(common_terms) / len(question_terms)
            
            # TF-IDF similarity score
            tfidf_similarity = self._calculate_tfidf_similarity(question, response)
            
            # Question type alignment score
            type_alignment = self._assess_question_type_fulfillment(question, response)
            
            # Weighted combination
            relevance = (term_overlap * 0.4) + (tfidf_similarity * 0.4) + (type_alignment * 0.2)
            
            return min(max(relevance, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Relevance calculation failed: {e}")
            return 0.5
    
    def _calculate_accuracy_score(self, response: str, sources: List[Dict]) -> float:
        """Estimate accuracy based on source content alignment"""
        try:
            if not sources:
                return 0.6  # Neutral score when no sources
            
            response_terms = set(self._extract_key_terms(response))
            if not response_terms:
                return 0.3  # Low score for empty response
            
            source_alignments = []
            
            for source in sources:
                source_content = source.get("content", "")
                if not source_content:
                    continue
                    
                source_terms = set(self._extract_key_terms(source_content))
                if source_terms:
                    # Calculate Jaccard similarity
                    intersection = len(response_terms & source_terms)
                    union = len(response_terms | source_terms)
                    jaccard = intersection / union if union > 0 else 0.0
                    source_alignments.append(jaccard)
            
            if source_alignments:
                # Use max alignment (best source match)
                accuracy = max(source_alignments)
            else:
                accuracy = 0.3
            
            return min(max(accuracy, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Accuracy calculation failed: {e}")
            return 0.5
    
    def _calculate_completeness_score(self, question: str, response: str) -> float:
        """Assess if response fully addresses the question"""
        try:
            # Response length assessment
            word_count = len(response.split())
            if word_count < 10:
                length_score = 0.3
            elif word_count < 30:
                length_score = 0.6
            elif word_count < 100:
                length_score = 0.9
            else:
                length_score = 1.0
            
            # Question type fulfillment
            type_score = self._assess_question_type_fulfillment(question, response)
            
            # Conclusiveness check
            conclusive_score = self._check_conclusive_language(response)
            
            # Weighted combination
            completeness = (length_score * 0.4) + (type_score * 0.4) + (conclusive_score * 0.2)
            
            return min(max(completeness, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Completeness calculation failed: {e}")
            return 0.5
    
    def _calculate_coherence_score(self, response: str) -> float:
        """Assess coherence and readability"""
        try:
            sentences = [s.strip() for s in self.sentence_pattern.split(response) if s.strip()]
            
            if len(sentences) <= 1:
                return 0.8  # Short responses are typically coherent
            
            # Transition words score
            transitions = [
                'however', 'therefore', 'moreover', 'furthermore', 'additionally',
                'consequently', 'meanwhile', 'similarly', 'in contrast', 'for example'
            ]
            
            response_lower = response.lower()
            transition_count = sum(1 for t in transitions if t in response_lower)
            transition_score = min(transition_count / len(sentences) * 2, 0.4)
            
            # Sentence length variation
            sentence_lengths = [len(s.split()) for s in sentences]
            if len(sentence_lengths) > 1:
                length_std = np.std(sentence_lengths)
                variation_score = min(length_std / 15, 0.3)
            else:
                variation_score = 0.1
            
            base_coherence = 0.5
            coherence = base_coherence + transition_score + variation_score
            
            return min(max(coherence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Coherence calculation failed: {e}")
            return 0.7
    
    def _calculate_citation_score(self, response: str, sources: List[Dict]) -> float:
        """Score source utilization"""
        try:
            if not sources:
                return 1.0  # Perfect if no sources expected
            
            response_lower = response.lower()
            citation_indicators = 0
            
            for source in sources:
                source_content = source.get("content", "")
                filename = source.get("filename", "")
                
                # Check for content references
                if source_content:
                    source_key_terms = self._extract_key_terms(source_content)[:5]
                    for term in source_key_terms:
                        if term.lower() in response_lower:
                            citation_indicators += 1
                            break
                
                # Check for document name references
                if filename:
                    doc_name = filename.replace('.pdf', '').replace('.txt', '').lower()
                    if doc_name in response_lower:
                        citation_indicators += 0.5
            
            citation_score = min(citation_indicators / len(sources), 1.0)
            return citation_score
            
        except Exception as e:
            logger.error(f"Citation calculation failed: {e}")
            return 0.5
    
    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF cosine similarity between two texts"""
        try:
            vectorizer = TfidfVectorizer(
                stop_words='english', 
                max_features=1000, 
                lowercase=True
            )
            
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.debug(f"TF-IDF similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall quality score"""
        weights = {
            "relevance": 0.30,
            "accuracy": 0.25,
            "completeness": 0.20,
            "coherence": 0.15,
            "citation": 0.10
        }
        
        overall = sum(scores.get(metric, 0.5) * weight for metric, weight in weights.items())
        return min(max(overall, 0.0), 1.0)
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract meaningful terms from text"""
        words = self.word_pattern.findall(text.lower())
        key_terms = [word for word in words if word not in self.stop_words and len(word) > 2]
        return key_terms[:20]
    
    def _assess_question_type_fulfillment(self, question: str, response: str) -> float:
        """Check if response type matches question type"""
        question_lower = question.lower()
        response_lower = response.lower()
        
        # What/Define questions
        if any(word in question_lower for word in ['what', 'define', 'explain']):
            if any(indicator in response_lower for indicator in ['is', 'refers to', 'means', 'defined as']):
                return 1.0
            return 0.6
        
        # How questions
        elif 'how' in question_lower:
            if any(indicator in response_lower for indicator in ['step', 'process', 'method', 'way']):
                return 1.0
            return 0.6
        
        # Why questions
        elif 'why' in question_lower:
            if any(indicator in response_lower for indicator in ['because', 'reason', 'due to']):
                return 1.0
            return 0.6
        
        return 0.8
    
    def _check_conclusive_language(self, response: str) -> float:
        """Check for conclusive language"""
        conclusive_phrases = [
            'in conclusion', 'therefore', 'thus', 'as a result',
            'consequently', 'in summary', 'overall', 'ultimately'
        ]
        
        response_lower = response.lower()
        conclusive_count = sum(1 for phrase in conclusive_phrases if phrase in response_lower)
        
        return min(conclusive_count * 0.3, 1.0)
