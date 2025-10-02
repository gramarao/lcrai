class QualityDashboard:
    def __init__(self):
        self.quality_attributes = {
            'relevance': {
                'weight': 0.25,
                'description': 'How well the response addresses the specific question',
                'calculation': 'Semantic similarity + keyword coverage + context alignment',
                'scale': '0.0-1.0 where 1.0 means perfectly relevant'
            },
            'accuracy': {
                'weight': 0.30,
                'description': 'Factual correctness based on source documents',
                'calculation': 'Source citation accuracy + fact verification + consistency check',
                'scale': '0.0-1.0 where 1.0 means completely accurate'
            },
            'completeness': {
                'weight': 0.20,
                'description': 'Coverage of all aspects mentioned in the question',
                'calculation': 'Question component coverage + information depth + missing element penalty',
                'scale': '0.0-1.0 where 1.0 means fully comprehensive'
            },
            'coherence': {
                'weight': 0.15,
                'description': 'Logical flow and readability of the response',
                'calculation': 'Sentence structure + transition quality + readability score',
                'scale': '0.0-1.0 where 1.0 means perfectly coherent'
            },
            'citation': {
                'weight': 0.10,
                'description': 'Proper attribution and source referencing',
                'calculation': 'Source mention accuracy + attribution completeness + reference format',
                'scale': '0.0-1.0 where 1.0 means properly cited'
            }
        }
    
    def calculate_weighted_score(self, scores: Dict[str, float]) -> Dict[str, any]:
        """Calculate overall quality score with detailed breakdown"""
        weighted_sum = 0.0
        component_scores = {}
        
        for attribute, config in self.quality_attributes.items():
            score = scores.get(attribute, 0.0)
            weight = config['weight']
            weighted_contribution = score * weight
            
            component_scores[attribute] = {
                'raw_score': score,
                'weight': weight,
                'weighted_contribution': weighted_contribution,
                'percentage': score * 100,
                'grade': self._score_to_grade(score)
            }
            
            weighted_sum += weighted_contribution
        
        return {
            'overall_score': weighted_sum,
            'overall_percentage': weighted_sum * 100,
            'overall_grade': self._score_to_grade(weighted_sum),
            'component_breakdown': component_scores,
            'calculation_formula': self._get_calculation_formula(),
            'improvement_suggestions': self._generate_improvement_suggestions(component_scores)
        }
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 0.95: return 'A+'
        elif score >= 0.90: return 'A'
        elif score >= 0.85: return 'A-'
        elif score >= 0.80: return 'B+'
        elif score >= 0.75: return 'B'
        elif score >= 0.70: return 'B-'
        elif score >= 0.65: return 'C+'
        elif score >= 0.60: return 'C'
        elif score >= 0.55: return 'C-'
        elif score >= 0.50: return 'D'
        else: return 'F'
    
    def _get_calculation_formula(self) -> str:
        """Return the weighted calculation formula"""
        formula_parts = []
        for attr, config in self.quality_attributes.items():
            formula_parts.append(f"{attr.title()}({config['weight']})")
        
        return f"Overall Score = {' + '.join(formula_parts)}"
    
    def _generate_improvement_suggestions(self, component_scores: Dict) -> List[str]:
        """Generate specific improvement suggestions based on scores"""
        suggestions = []
        
        for attribute, scores in component_scores.items():
            if scores['raw_score'] < 0.7:  # Below B- grade
                if attribute == 'relevance':
                    suggestions.append("Improve relevance by better understanding question intent and focusing on specific topics asked")
                elif attribute == 'accuracy':
                    suggestions.append("Enhance accuracy by more careful source verification and fact-checking against authoritative documents")
                elif attribute == 'completeness':
                    suggestions.append("Increase completeness by addressing all aspects of multi-part questions and providing comprehensive coverage")
                elif attribute == 'coherence':
                    suggestions.append("Improve coherence through better paragraph structure, clearer transitions, and logical information flow")
                elif attribute == 'citation':
                    suggestions.append("Strengthen citations by properly attributing information to specific sources and documents")
        
        if not suggestions:
            suggestions.append("Excellent performance across all quality dimensions - maintain current standards")
        
        return suggestions