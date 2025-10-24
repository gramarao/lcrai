# Create file: app/api/quality_routes.py

from fastapi import APIRouter, HTTPException, Depends, Body
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from app.models.database import get_db, UserFeedback
from app.services.quality_dashboard import QualityDashboard

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/quality", tags=["quality"])

# Initialize quality dashboard
quality_dashboard = QualityDashboard()

@router.get("/attributes")
async def get_quality_attributes():
    """Get all quality attribute definitions and their weights"""
    try:
        return {
            "quality_attributes": quality_dashboard.quality_attributes,
            "total_weight": sum(attr['weight'] for attr in quality_dashboard.quality_attributes.values()),
            "calculation_method": "Weighted average of all component scores"
        }
    except Exception as e:
        logger.exception("Failed to get quality attributes")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze")
async def analyze_quality_scores(
    scores: Dict[str, float] = Body(..., example={
        "relevance": 0.85,
        "accuracy": 0.92,
        "completeness": 0.78,
        "coherence": 0.88,
        "citation": 0.75
    })
):
    """Analyze quality scores and get detailed breakdown"""
    try:
        # Validate scores
        for attribute in quality_dashboard.quality_attributes.keys():
            if attribute not in scores:
                raise HTTPException(status_code=400, detail=f"Missing score for {attribute}")
            if not 0.0 <= scores[attribute] <= 1.0:
                raise HTTPException(status_code=400, detail=f"Score for {attribute} must be between 0.0 and 1.0")
        
        analysis = quality_dashboard.calculate_weighted_score(scores)
        
        return {
            "input_scores": scores,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Quality analysis failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/{query_id}")
async def get_quality_dashboard_for_query(query_id: str, db: Session = Depends(get_db)):
    """Get quality dashboard data for a specific query"""
    try:
        # Get feedback for this query
        feedback = db.query(UserFeedback).filter(UserFeedback.query_id == query_id).first()
        
        if not feedback:
            raise HTTPException(status_code=404, detail="Query not found")
        
        # Get quality scores
        scores = {
            'relevance': feedback.relevance_score or 0.5,
            'accuracy': feedback.accuracy_score or 0.5,
            'completeness': feedback.completeness_score or 0.5,
            'coherence': feedback.coherence_score or 0.5,
            'citation': feedback.citation_score or 0.5
        }
        
        # Analyze with dashboard
        analysis = quality_dashboard.calculate_weighted_score(scores)
        
        return {
            "query_id": query_id,
            "question": feedback.question,
            "response": feedback.response,
            "human_rating": feedback.rating,
            "quality_analysis": analysis,
            "created_at": feedback.created_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get dashboard for query {query_id}")
        raise HTTPException(status_code=500, detail=str(e))
