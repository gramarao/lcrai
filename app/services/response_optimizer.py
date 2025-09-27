import logging
from typing import List, Dict, Tuple
from sqlalchemy.orm import Session
from app.models.database import UserFeedback, QueryPerformance, DocumentChunk, Document
from app.services.rag_service_googleai import rag_service
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class ResponseOptimizer:
    def __init__(self):
        self.learning_rate = 0.1
        self.min_feedback_threshold = 5  # Minimum feedback needed for optimization
        
    async def optimize_retrieval_for_query(self, db: Session, query_pattern: str) -> Dict:
        """Optimize document retrieval based on feedback for similar queries"""
        try:
            # Get feedback for similar query patterns
            feedback_data = db.query(UserFeedback).join(
                QueryPerformance, UserFeedback.query_id == QueryPerformance.query_hash
            ).filter(
                QueryPerformance.query_pattern == query_pattern,
                UserFeedback.rating.is_not(None)
            ).all()
            
            if len(feedback_data) < self.min_feedback_threshold:
                logger.info(f"Insufficient feedback for optimization: {len(feedback_data)} samples")
                return {"status": "insufficient_data", "samples": len(feedback_data)}
            
            # Analyze which documents/chunks performed best
            document_performance = defaultdict(list)
            chunk_performance = defaultdict(list)
            
            for feedback in feedback_data:
                # Get associated sources for this feedback
                sources = db.query(FeedbackSource).filter(
                    FeedbackSource.feedback_id == feedback.id
                ).all()
                
                for source in sources:
                    document_performance[source.document_id].append(feedback.rating)
                    chunk_performance[source.chunk_id].append(feedback.rating)
            
            # Calculate average performance scores
            doc_scores = {}
            for doc_id, ratings in document_performance.items():
                doc_scores[doc_id] = {
                    "avg_rating": np.mean(ratings),
                    "sample_count": len(ratings),
                    "confidence": min(len(ratings) / 10.0, 1.0)  # Confidence based on sample size
                }
            
            chunk_scores = {}
            for chunk_id, ratings in chunk_performance.items():
                chunk_scores[chunk_id] = {
                    "avg_rating": np.mean(ratings),
                    "sample_count": len(ratings),
                    "confidence": min(len(ratings) / 5.0, 1.0)
                }
            
            # Update QueryPerformance with optimization data
            performance_record = db.query(QueryPerformance).filter(
                QueryPerformance.query_pattern == query_pattern
            ).first()
            
            if performance_record:
                # Store best performing documents
                best_docs = sorted(doc_scores.items(), 
                                 key=lambda x: x[1]["avg_rating"] * x[1]["confidence"], 
                                 reverse=True)[:5]
                
                performance_record.best_documents = [doc_id for doc_id, _ in best_docs]
                
                # Optimize source count based on feedback
                optimal_sources = self._calculate_optimal_source_count(feedback_data)
                performance_record.optimal_sources_count = optimal_sources
                
                db.commit()
            
            return {
                "status": "optimized",
                "query_pattern": query_pattern,
                "samples_analyzed": len(feedback_data),
                "top_documents": doc_scores,
                "optimization_applied": True
            }
            
        except Exception as e:
            logger.error(f"Query optimization failed: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    async def enhance_query_expansion(self, db: Session, original_query: str) -> List[str]:
        """Generate query variations based on successful similar queries"""
        try:
            # Find successful queries with high ratings
            successful_queries = db.query(UserFeedback).filter(
                UserFeedback.rating >= 4,
                UserFeedback.overall_quality >= 0.7
            ).order_by(UserFeedback.created_at.desc()).limit(50).all()
            
            # Extract successful query patterns
            query_terms = set(original_query.lower().split())
            similar_queries = []
            
            for feedback in successful_queries:
                feedback_terms = set(feedback.question.lower().split())
                overlap = len(query_terms & feedback_terms)
                
                if overlap >= 2:  # At least 2 terms in common
                    similar_queries.append(feedback.question)
            
            # Generate expanded queries
            expanded_queries = [original_query]  # Always include original
            
            if similar_queries:
                # Extract common successful patterns
                for similar in similar_queries[:3]:  # Top 3 similar successful queries
                    # Simple expansion: combine terms
                    combined_terms = set(original_query.split()) | set(similar.split())
                    if len(combined_terms) <= len(original_query.split()) + 3:  # Don't make too long
                        expanded_query = " ".join(combined_terms)
                        expanded_queries.append(expanded_query)
            
            return expanded_queries[:3]  # Return top 3 expansions
            
        except Exception as e:
            logger.error(f"Query expansion failed: {str(e)}")
            return [original_query]
    
    async def adjust_chunk_ranking(self, db: Session, chunks: List[Dict], query_pattern: str) -> List[Dict]:
        """Adjust chunk ranking based on historical performance"""
        try:
            # Get performance data for this query pattern
            performance_data = db.query(QueryPerformance).filter(
                QueryPerformance.query_pattern == query_pattern
            ).first()
            
            if not performance_data or not performance_data.best_documents:
                return chunks  # No optimization data available
            
            # Create ranking boost for high-performing documents
            best_docs = set(performance_data.best_documents)
            
            # Adjust chunk scores based on document performance
            for chunk in chunks:
                if chunk.get("document_id") in best_docs:
                    # Boost score for chunks from high-performing documents
                    original_score = chunk.get("similarity_score", 0.5)
                    chunk["similarity_score"] = min(original_score * 1.2, 1.0)
                    chunk["boosted"] = True
                else:
                    chunk["boosted"] = False
            
            # Re-sort chunks by adjusted scores
            chunks.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            
            logger.info(f"Adjusted ranking for {len(chunks)} chunks based on {query_pattern} performance")
            return chunks
            
        except Exception as e:
            logger.error(f"Chunk ranking adjustment failed: {str(e)}")
            return chunks
    
    def _calculate_optimal_source_count(self, feedback_data: List[UserFeedback]) -> int:
        """Calculate optimal number of sources based on feedback"""
        try:
            # Analyze relationship between source count and rating
            source_count_ratings = defaultdict(list)
            
            for feedback in feedback_data:
                if feedback.sources_count and feedback.rating:
                    source_count_ratings[feedback.sources_count].append(feedback.rating)
            
            if not source_count_ratings:
                return 3  # Default
            
            # Find source count with highest average rating
            best_count = 3
            best_avg_rating = 0
            
            for count, ratings in source_count_ratings.items():
                if len(ratings) >= 3:  # Need at least 3 samples
                    avg_rating = np.mean(ratings)
                    if avg_rating > best_avg_rating:
                        best_avg_rating = avg_rating
                        best_count = count
            
            return max(min(best_count, 10), 1)  # Clamp between 1 and 10
            
        except Exception as e:
            logger.error(f"Optimal source count calculation failed: {str(e)}")
            return 3

# Initialize response optimizer
response_optimizer = ResponseOptimizer()