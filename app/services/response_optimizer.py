import logging
from typing import List, Dict, Tuple
from sqlalchemy.orm import Session
from app.models.database import UserFeedback, QueryPerformance, FeedbackSource, DocumentChunk, Document
import numpy as np
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)

class ResponseOptimizer:
    def __init__(self):
        self.learning_rate = 0.1
        self.min_feedback_threshold = 5
        logger.info("ResponseOptimizer initialized")
        
    async def optimize_retrieval_for_query(self, db: Session, query_pattern: str) -> Dict:
        """Optimize document retrieval based on feedback for similar queries"""
        try:
            feedback_data = db.query(UserFeedback).join(
                QueryPerformance, UserFeedback.query_id == QueryPerformance.query_hash
            ).filter(
                QueryPerformance.query_pattern == query_pattern,
                UserFeedback.rating.is_not(None)
            ).all()
            
            if len(feedback_data) < self.min_feedback_threshold:
                logger.info(f"Insufficient feedback for optimization: {len(feedback_data)} samples")
                return {"status": "insufficient_data", "samples": len(feedback_data)}
            
            document_performance = defaultdict(list)
            chunk_performance = defaultdict(list)
            
            for feedback in feedback_data:
                sources = db.query(FeedbackSource).filter(
                    FeedbackSource.feedback_id == feedback.id
                ).all()
                
                for source in sources:
                    document_performance[str(source.document_id)].append(feedback.rating)
                    chunk_performance[str(source.chunk_id)].append(feedback.rating)
            
            doc_scores = {}
            for doc_id, ratings in document_performance.items():
                doc_scores[doc_id] = {
                    "avg_rating": float(np.mean(ratings)),
                    "sample_count": len(ratings),
                    "confidence": min(len(ratings) / 10.0, 1.0)
                }
            
            chunk_scores = {}
            for chunk_id, ratings in chunk_performance.items():
                chunk_scores[chunk_id] = {
                    "avg_rating": float(np.mean(ratings)),
                    "sample_count": len(ratings),
                    "confidence": min(len(ratings) / 5.0, 1.0)
                }
            
            performance_record = db.query(QueryPerformance).filter(
                QueryPerformance.query_pattern == query_pattern
            ).first()
            
            if performance_record:
                best_docs = sorted(
                    doc_scores.items(),
                    key=lambda x: x[1]["avg_rating"] * x[1]["confidence"],
                    reverse=True
                )[:5]
                
                # Convert string IDs back to UUIDs for database storage
                performance_record.best_documents = [uuid.UUID(doc_id) for doc_id, _ in best_docs]
                performance_record.optimal_sources_count = self._calculate_optimal_source_count(feedback_data)
                db.commit()
                logger.info(f"Optimization applied for query pattern: {query_pattern}")
            
            return {
                "status": "optimized",
                "query_pattern": query_pattern,
                "samples_analyzed": len(feedback_data),
                "top_documents": doc_scores,
                "optimization_applied": True
            }
            
        except Exception as e:
            logger.exception(f"Query optimization failed for pattern: {query_pattern}")
            return {"status": "error", "error": str(e)}
    
    async def enhance_query_expansion(self, db: Session, original_query: str) -> List[str]:
        """Generate query variations based on successful similar queries"""
        try:
            successful_queries = db.query(UserFeedback).filter(
                UserFeedback.rating >= 4,
                UserFeedback.overall_quality_score >= 0.7
            ).order_by(UserFeedback.created_at.desc()).limit(50).all()
            
            query_terms = set(original_query.lower().split())
            similar_queries = []
            
            for feedback in successful_queries:
                feedback_terms = set(feedback.question.lower().split())
                overlap = len(query_terms & feedback_terms)
                
                if overlap >= 2:
                    similar_queries.append(feedback.question)
            
            expanded_queries = [original_query]
            
            if similar_queries:
                for similar in similar_queries[:3]:
                    combined_terms = set(original_query.split()) | set(similar.split())
                    if len(combined_terms) <= len(original_query.split()) + 3:
                        expanded_query = " ".join(combined_terms)
                        expanded_queries.append(expanded_query)
            
            logger.info(f"Enhanced query expansion for '{original_query[:50]}...'")
            return expanded_queries[:3]
            
        except Exception as e:
            logger.exception(f"Query expansion failed for query: {original_query[:50]}...")
            return [original_query]
    
    async def adjust_chunk_ranking(self, db: Session, chunks: List[Dict], query_pattern: str) -> List[Dict]:
        """Adjust chunk ranking based on historical performance"""
        try:
            performance_data = db.query(QueryPerformance).filter(
                QueryPerformance.query_pattern == query_pattern
            ).first()
            
            if not performance_data or not performance_data.best_documents:
                logger.info(f"No optimization data for chunk ranking for pattern: {query_pattern}")
                return chunks
            
            best_docs = {str(doc_id) for doc_id in performance_data.best_documents}
            
            for chunk in chunks:
                if str(chunk.get("document_id")) in best_docs:
                    original_score = chunk.get("similarity_score", 0.5)
                    chunk["similarity_score"] = min(original_score * 1.2, 1.0)
                    chunk["boosted"] = True
                else:
                    chunk["boosted"] = False
            
            chunks.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            logger.info(f"Adjusted ranking for {len(chunks)} chunks based on {query_pattern} performance")
            return chunks
            
        except Exception as e:
            logger.exception(f"Chunk ranking adjustment failed for pattern: {query_pattern}")
            return chunks
    
    def _calculate_optimal_source_count(self, feedback_data: List[UserFeedback]) -> int:
        """Calculate optimal number of sources based on feedback"""
        try:
            source_count_ratings = defaultdict(list)
            
            for feedback in feedback_data:
                if feedback.sources_count is not None and feedback.rating is not None:
                    source_count_ratings[feedback.sources_count].append(feedback.rating)
            
            if not source_count_ratings:
                return 3
            
            best_count = 3
            best_avg_rating = 0.0
            
            for count, ratings in source_count_ratings.items():
                if len(ratings) >= 3:
                    avg_rating = float(np.mean(ratings))
                    if avg_rating > best_avg_rating:
                        best_avg_rating = avg_rating
                        best_count = count
            
            logger.info(f"Calculated optimal source count: {best_count}")
            return max(min(best_count, 10), 1)
            
        except Exception as e:
            logger.exception("Optimal source count calculation failed")
            return 3

# No global instance created here - will be instantiated in RAGService
