#!/usr/bin/env python3
"""
Clinical Decision Support System
AI-powered decision assistance for clinical robot operations.

This module implements:
- Clinical knowledge base and reasoning
- Medication interaction checking
- Patient-specific recommendations
- Risk assessment and mitigation
- Clinical guideline adherence
- Decision explanation and visualization
- Real-time decision support
- Integration with clinical workflows

Author: Idriss Djiofack Teledjieu
Clinical Robot Adaptation Project
HIRO Laboratory, University of Colorado Boulder
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import warnings

# Machine learning and AI
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

# Knowledge representation
import networkx as nx
from rdflib import Graph, URIRef, Literal, Namespace

# Clinical data
import pydicom
from pydantic import BaseModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clinical_decision_support.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class DecisionType(Enum):
    """Types of clinical decisions."""
    MEDICATION_SELECTION = "medication_selection"
    DOSAGE_CALCULATION = "dosage_calculation"
    INTERACTION_CHECK = "interaction_check"
    RISK_ASSESSMENT = "risk_assessment"
    SAFETY_RECOMMENDATION = "safety_recommendation"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"

class ConfidenceLevel(Enum):
    """Confidence levels for decisions."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"

class UrgencyLevel(Enum):
    """Urgency levels for decisions."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class Decision:
    """Clinical decision recommendation."""
    id: str
    decision_type: DecisionType
    recommendation: str
    confidence: ConfidenceLevel
    urgency: UrgencyLevel
    rationale: List[str]
    alternatives: List[str]
    risks: List[str]
    benefits: List[str]
    supporting_evidence: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'decision_type': self.decision_type.value,
            'recommendation': self.recommendation,
            'confidence': self.confidence.value,
            'urgency': self.urgency.value,
            'rationale': self.rationale,
            'alternatives': self.alternatives,
            'risks': self.risks,
            'benefits': self.benefits,
            'supporting_evidence': self.supporting_evidence,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

@dataclass
class PatientProfile:
    """Patient clinical profile."""
    patient_id: str
    age: int
    weight: float  # kg
    height: float  # cm
    gender: str
    allergies: List[str]
    current_medications: List[str]
    medical_conditions: List[str]
    vital_signs: Dict[str, float]
    lab_results: Dict[str, float]
    contraindications: List[str]
    preferences: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'patient_id': self.patient_id,
            'age': self.age,
            'weight': self.weight,
            'height': self.height,
            'gender': self.gender,
            'allergies': self.allergies,
            'current_medications': self.current_medications,
            'medical_conditions': self.medical_conditions,
            'vital_signs': self.vital_signs,
            'lab_results': self.lab_results,
            'contraindications': self.contraindications,
            'preferences': self.preferences
        }

class ClinicalKnowledgeBase:
    """Clinical knowledge base for decision support."""
    
    def __init__(self, knowledge_path: str = "./clinical_knowledge"):
        self.knowledge_path = Path(knowledge_path)
        self.knowledge_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize knowledge graphs
        self.medication_graph = nx.DiGraph()
        self.condition_graph = nx.DiGraph()
        self.interaction_graph = nx.DiGraph()
        
        # Load knowledge
        self._load_medication_knowledge()
        self._load_interaction_knowledge()
        self._load_clinical_guidelines()
        
        logger.info("Clinical Knowledge Base initialized")
    
    def _load_medication_knowledge(self):
        """Load medication knowledge."""
        # Default medication knowledge
        medications = {
            'aspirin': {
                'type': 'nsaid',
                'indications': ['pain', 'fever', 'inflammation'],
                'contraindications': ['bleeding_disorder', 'ulcer'],
                'side_effects': ['bleeding', 'stomach_upset'],
                'dosage_range': (50, 500)
            },
            'ibuprofen': {
                'type': 'nsaid',
                'indications': ['pain', 'fever', 'inflammation'],
                'contraindications': ['kidney_disease', 'ulcer'],
                'side_effects': ['kidney_damage', 'stomach_upset'],
                'dosage_range': (100, 800)
            },
            'acetaminophen': {
                'type': 'analgesic',
                'indications': ['pain', 'fever'],
                'contraindications': ['liver_disease'],
                'side_effects': ['liver_damage'],
                'dosage_range': (325, 1000)
            },
            'metformin': {
                'type': 'antidiabetic',
                'indications': ['diabetes_type2'],
                'contraindications': ['kidney_disease', 'liver_disease'],
                'side_effects': ['lactic_acidosis', 'gastrointestinal'],
                'dosage_range': (500, 2000)
            },
            'lisinopril': {
                'type': 'ace_inhibitor',
                'indications': ['hypertension', 'heart_failure'],
                'contraindications': ['pregnancy', 'angioedema'],
                'side_effects': ['cough', 'hypotension'],
                'dosage_range': (5, 40)
            }
        }
        
        # Build medication graph
        for med_name, med_info in medications.items():
            self.medication_graph.add_node(med_name, **med_info)
        
        logger.info(f"Loaded {len(medications)} medications")
    
    def _load_interaction_knowledge(self):
        """Load drug interaction knowledge."""
        # Default interaction knowledge
        interactions = [
            ('aspirin', 'ibuprofen', 'major', 'increased_bleeding_risk'),
            ('aspirin', 'lisinopril', 'moderate', 'reduced_effectiveness'),
            ('ibuprofen', 'lisinopril', 'moderate', 'reduced_effectiveness'),
            ('metformin', 'lisinopril', 'minor', 'increased_lactic_acidosis_risk'),
            ('acetaminophen', 'alcohol', 'major', 'increased_liver_damage_risk')
        ]
        
        for med1, med2, severity, effect in interactions:
            self.interaction_graph.add_edge(med1, med2, severity=severity, effect=effect)
        
        logger.info(f"Loaded {len(interactions)} drug interactions")
    
    def _load_clinical_guidelines(self):
        """Load clinical guidelines."""
        # Default guidelines
        self.guidelines = {
            'pain_management': {
                'first_line': ['acetaminophen'],
                'second_line': ['nsaid'],
                'contraindications_check': True,
                'age_adjustments': True
            },
            'hypertension': {
                'first_line': ['ace_inhibitor'],
                'target_bp': (120, 80),
                'monitoring_required': True
            },
            'diabetes': {
                'first_line': ['metformin'],
                'target_hba1c': 7.0,
                'kidney_monitoring': True
            }
        }
        
        logger.info(f"Loaded {len(self.guidelines)} clinical guidelines")
    
    def get_medication_info(self, medication: str) -> Optional[Dict[str, Any]]:
        """Get information about a medication."""
        if medication in self.medication_graph:
            return self.medication_graph.nodes[medication]
        return None
    
    def check_interaction(self, med1: str, med2: str) -> Optional[Dict[str, Any]]:
        """Check for drug interaction."""
        if self.interaction_graph.has_edge(med1, med2):
            return self.interaction_graph.edges[med1, med2]
        elif self.interaction_graph.has_edge(med2, med1):
            return self.interaction_graph.edges[med2, med1]
        return None
    
    def get_guideline(self, condition: str) -> Optional[Dict[str, Any]]:
        """Get clinical guideline for a condition."""
        return self.guidelines.get(condition)

class RiskAssessmentModel(nn.Module):
    """Neural network for risk assessment."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU()
        )
        
        self.risk_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.urgency_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),  # 4 urgency levels
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        features = self.encoder(x)
        risk = self.risk_head(features)
        urgency = self.urgency_head(features)
        return risk, urgency

class ClinicalDecisionSupportSystem:
    """Main clinical decision support system."""
    
    def __init__(self, knowledge_path: str = "./clinical_knowledge"):
        self.knowledge_base = ClinicalKnowledgeBase(knowledge_path)
        
        # Initialize ML models
        self.risk_model = None
        self.scaler = StandardScaler()
        
        # Decision history
        self.decision_history = []
        
        # Patient profiles
        self.patient_profiles = {}
        
        logger.info("Clinical Decision Support System initialized")
    
    def initialize_risk_model(self, training_data: pd.DataFrame):
        """Initialize and train risk assessment model."""
        try:
            # Prepare features
            feature_columns = ['age', 'weight', 'vital_sign_1', 'vital_sign_2', 'lab_value_1', 'lab_value_2']
            X = training_data[feature_columns].values
            y = training_data['risk_label'].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y).unsqueeze(1)
            
            # Create model
            input_dim = X_scaled.shape[1]
            self.risk_model = RiskAssessmentModel(input_dim)
            
            # Train model
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(self.risk_model.parameters(), lr=0.001)
            
            for epoch in range(100):
                optimizer.zero_grad()
                risk_pred, _ = self.risk_model(X_tensor)
                loss = criterion(risk_pred, y_tensor)
                loss.backward()
                optimizer.step()
            
            logger.info("Risk assessment model trained successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize risk model: {e}")
    
    def add_patient_profile(self, profile: PatientProfile):
        """Add patient profile to system."""
        self.patient_profiles[profile.patient_id] = profile
        logger.info(f"Patient profile added: {profile.patient_id}")
    
    def get_patient_profile(self, patient_id: str) -> Optional[PatientProfile]:
        """Get patient profile."""
        return self.patient_profiles.get(patient_id)
    
    def check_medication_interactions(self, patient_id: str, 
                                     new_medication: str) -> Decision:
        """Check for medication interactions for a patient."""
        patient = self.get_patient_profile(patient_id)
        if not patient:
            return self._create_error_decision(
                DecisionType.INTERACTION_CHECK,
                f"Patient profile not found: {patient_id}"
            )
        
        interactions = []
        high_risk_interactions = []
        
        # Check interactions with current medications
        for current_med in patient.current_medications:
            interaction = self.knowledge_base.check_interaction(current_med, new_medication)
            if interaction:
                interactions.append({
                    'medication': current_med,
                    'severity': interaction['severity'],
                    'effect': interaction['effect']
                })
                
                if interaction['severity'] == 'major':
                    high_risk_interactions.append(current_med)
        
        # Check allergies
        med_info = self.knowledge_base.get_medication_info(new_medication)
        if med_info:
            for allergy in patient.allergies:
                if allergy.lower() in new_medication.lower():
                    interactions.append({
                        'medication': new_medication,
                        'severity': 'major',
                        'effect': 'allergic_reaction'
                    })
                    high_risk_interactions.append(new_medication)
        
        # Create decision
        if high_risk_interactions:
            recommendation = f"Do not prescribe {new_medication} due to high-risk interactions"
            confidence = ConfidenceLevel.HIGH
            urgency = UrgencyLevel.CRITICAL
        elif interactions:
            recommendation = f"Prescribe {new_medication} with caution - monitor for interactions"
            confidence = ConfidenceLevel.MEDIUM
            urgency = UrgencyLevel.HIGH
        else:
            recommendation = f"{new_medication} can be prescribed safely"
            confidence = ConfidenceLevel.HIGH
            urgency = UrgencyLevel.LOW
        
        decision = Decision(
            id=f"interaction_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            decision_type=DecisionType.INTERACTION_CHECK,
            recommendation=recommendation,
            confidence=confidence,
            urgency=urgency,
            rationale=self._generate_interaction_rationale(interactions),
            alternatives=self._generate_interaction_alternatives(new_medication, high_risk_interactions),
            risks=[f"Interaction with {i['medication']}: {i['effect']}" for i in interactions],
            benefits=["Effective treatment for condition"],
            supporting_evidence=[
                {
                    'source': 'drug_interaction_database',
                    'evidence': str(interactions)
                }
            ],
            metadata={
                'patient_id': patient_id,
                'new_medication': new_medication,
                'interactions': interactions
            }
        )
        
        self.decision_history.append(decision)
        return decision
    
    def _generate_interaction_rationale(self, interactions: List[Dict[str, Any]]) -> List[str]:
        """Generate rationale for interaction decision."""
        rationale = []
        
        if not interactions:
            rationale.append("No known drug interactions detected")
            return rationale
        
        for interaction in interactions:
            rationale.append(
                f"Interaction with {interaction['medication']}: {interaction['effect']} "
                f"(severity: {interaction['severity']})"
            )
        
        return rationale
    
    def _generate_interaction_alternatives(self, medication: str, 
                                          high_risk_meds: List[str]) -> List[str]:
        """Generate alternative medications."""
        alternatives = []
        
        # Get medication type
        med_info = self.knowledge_base.get_medication_info(medication)
        if med_info:
            med_type = med_info.get('type')
            
            # Find alternatives of same type
            for med_name in self.knowledge_base.medication_graph.nodes():
                if med_name != medication:
                    alt_info = self.knowledge_base.get_medication_info(med_name)
                    if alt_info and alt_info.get('type') == med_type:
                        alternatives.append(med_name)
        
        return alternatives[:5]  # Return top 5 alternatives
    
    def assess_medication_risk(self, patient_id: str, 
                              medication: str) -> Decision:
        """Assess risk of medication for patient."""
        patient = self.get_patient_profile(patient_id)
        if not patient:
            return self._create_error_decision(
                DecisionType.RISK_ASSESSMENT,
                f"Patient profile not found: {patient_id}"
            )
        
        # Get medication info
        med_info = self.knowledge_base.get_medication_info(medication)
        if not med_info:
            return self._create_error_decision(
                DecisionType.RISK_ASSESSMENT,
                f"Medication not found: {medication}"
            )
        
        # Check contraindications
        contraindications = []
        for condition in patient.medical_conditions:
            if condition in med_info.get('contraindications', []):
                contraindications.append(condition)
        
        # Check allergies
        allergic = False
        for allergy in patient.allergies:
            if allergy.lower() in medication.lower():
                allergic = True
                contraindications.append(f"Allergy to {allergy}")
        
        # Assess risk level
        if contraindications:
            risk_level = "high"
            recommendation = f"{medication} contraindicated for this patient"
            confidence = ConfidenceLevel.HIGH
            urgency = UrgencyLevel.CRITICAL
        elif allergic:
            risk_level = "high"
            recommendation = f"{medication} contraindicated due to allergy"
            confidence = ConfidenceLevel.HIGH
            urgency = UrgencyLevel.CRITICAL
        else:
            risk_level = "low"
            recommendation = f"{medication} can be safely prescribed"
            confidence = ConfidenceLevel.HIGH
            urgency = UrgencyLevel.LOW
        
        decision = Decision(
            id=f"risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            decision_type=DecisionType.RISK_ASSESSMENT,
            recommendation=recommendation,
            confidence=confidence,
            urgency=urgency,
            rationale=[
                f"Patient conditions: {', '.join(patient.medical_conditions)}",
                f"Medication contraindications: {', '.join(med_info.get('contraindications', []))}",
                f"Allergies: {', '.join(patient.allergies)}"
            ],
            alternatives=self._generate_interaction_alternatives(medication, []),
            risks=[f"Contraindication: {c}" for c in contraindications],
            benefits=med_info.get('indications', []),
            supporting_evidence=[
                {
                    'source': 'clinical_knowledge_base',
                    'evidence': med_info
                }
            ],
            metadata={
                'patient_id': patient_id,
                'medication': medication,
                'risk_level': risk_level,
                'contraindications': contraindications
            }
        )
        
        self.decision_history.append(decision)
        return decision
    
    def recommend_medication(self, patient_id: str, condition: str) -> Decision:
        """Recommend medication for patient condition."""
        patient = self.get_patient_profile(patient_id)
        if not patient:
            return self._create_error_decision(
                DecisionType.MEDICATION_SELECTION,
                f"Patient profile not found: {patient_id}"
            )
        
        # Get guideline for condition
        guideline = self.knowledge_base.get_guideline(condition)
        if not guideline:
            return self._create_error_decision(
                DecisionType.MEDICATION_SELECTION,
                f"No guideline found for condition: {condition}"
            )
        
        # Get first-line medications
        first_line_meds = guideline.get('first_line', [])
        
        # Filter out contraindicated medications
        recommended_meds = []
        for med in first_line_meds:
            med_info = self.knowledge_base.get_medication_info(med)
            if med_info:
                # Check contraindications
                contraindicated = False
                for condition in patient.medical_conditions:
                    if condition in med_info.get('contraindications', []):
                        contraindicated = True
                        break
                
                # Check allergies
                for allergy in patient.allergies:
                    if allergy.lower() in med.lower():
                        contraindicated = True
                        break
                
                if not contraindicated:
                    recommended_meds.append(med)
        
        if recommended_meds:
            recommendation = f"Recommended medications: {', '.join(recommended_meds)}"
            confidence = ConfidenceLevel.HIGH
            urgency = UrgencyLevel.MEDIUM
        else:
            recommendation = "No first-line medications suitable - consider specialist consultation"
            confidence = ConfidenceLevel.MEDIUM
            urgency = UrgencyLevel.HIGH
        
        decision = Decision(
            id=f"medication_recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            decision_type=DecisionType.MEDICATION_SELECTION,
            recommendation=recommendation,
            confidence=confidence,
            urgency=urgency,
            rationale=[
                f"Condition: {condition}",
                f"First-line medications: {', '.join(first_line_meds)}",
                f"Patient contraindications: {', '.join(patient.contraindications)}",
                f"Patient allergies: {', '.join(patient.allergies)}"
            ],
            alternatives=first_line_meds,
            risks=["Potential side effects", "Drug interactions"],
            benefits=["Evidence-based treatment", "Standard of care"],
            supporting_evidence=[
                {
                    'source': 'clinical_guideline',
                    'guideline': guideline
                }
            ],
            metadata={
                'patient_id': patient_id,
                'condition': condition,
                'recommended_medications': recommended_meds
            }
        )
        
        self.decision_history.append(decision)
        return decision
    
    def calculate_dosage(self, patient_id: str, medication: str) -> Decision:
        """Calculate appropriate dosage for patient."""
        patient = self.get_patient_profile(patient_id)
        if not patient:
            return self._create_error_decision(
                DecisionType.DOSAGE_CALCULATION,
                f"Patient profile not found: {patient_id}"
            )
        
        # Get medication info
        med_info = self.knowledge_base.get_medication_info(medication)
        if not med_info:
            return self._create_error_decision(
                DecisionType.DOSAGE_CALCULATION,
                f"Medication not found: {medication}"
            )
        
        # Calculate dosage based on weight
        min_dose, max_dose = med_info.get('dosage_range', (0, 0))
        
        # Weight-based calculation (simplified)
        weight_based_dose = patient.weight * 10  # mg/kg
        calculated_dose = max(min_dose, min(weight_based_dose, max_dose))
        
        # Age adjustment
        if patient.age > 65:
            calculated_dose *= 0.8  # Reduce dose for elderly
        
        # Round to nearest standard dose
        calculated_dose = round(calculated_dose / 25) * 25
        
        recommendation = f"Recommended dosage: {calculated_dose} mg"
        confidence = ConfidenceLevel.MEDIUM
        urgency = UrgencyLevel.LOW
        
        decision = Decision(
            id=f"dosage_calculation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            decision_type=DecisionType.DOSAGE_CALCULATION,
            recommendation=recommendation,
            confidence=confidence,
            urgency=urgency,
            rationale=[
                f"Patient weight: {patient.weight} kg",
                f"Patient age: {patient.age} years",
                f"Standard dosage range: {min_dose}-{max_dose} mg",
                f"Age adjustment applied: {'Yes' if patient.age > 65 else 'No'}"
            ],
            alternatives=[
                f"{calculated_dose - 25} mg",
                f"{calculated_dose + 25} mg"
            ],
            risks=["Overdose", "Underdose"],
            benefits=["Optimal therapeutic effect", "Minimized side effects"],
            supporting_evidence=[
                {
                    'source': 'dosing_guidelines',
                    'calculation': f"weight_based: {weight_based_dose:.1f} mg, adjusted: {calculated_dose} mg"
                }
            ],
            metadata={
                'patient_id': patient_id,
                'medication': medication,
                'calculated_dosage': calculated_dose,
                'weight_based_dose': weight_based_dose
            }
        )
        
        self.decision_history.append(decision)
        return decision
    
    def _create_error_decision(self, decision_type: DecisionType, 
                              error_message: str) -> Decision:
        """Create error decision."""
        return Decision(
            id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            decision_type=decision_type,
            recommendation=f"Error: {error_message}",
            confidence=ConfidenceLevel.UNCERTAIN,
            urgency=UrgencyLevel.HIGH,
            rationale=[error_message],
            alternatives=[],
            risks=[],
            benefits=[],
            supporting_evidence=[],
            metadata={'error': error_message}
        )
    
    def get_decision_history(self, patient_id: str = None, 
                           decision_type: DecisionType = None) -> List[Decision]:
        """Get decision history, optionally filtered."""
        history = self.decision_history
        
        if patient_id:
            history = [d for d in history if d.metadata.get('patient_id') == patient_id]
        
        if decision_type:
            history = [d for d in history if d.decision_type == decision_type]
        
        return history
    
    def generate_decision_report(self, decision: Decision) -> str:
        """Generate human-readable decision report."""
        report = f"""
# Clinical Decision Support Report

**Decision ID:** {decision.id}
**Decision Type:** {decision.decision_type.value}
**Timestamp:** {decision.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Recommendation
{decision.recommendation}

**Confidence Level:** {decision.confidence.value.upper()}
**Urgency Level:** {decision.urgency.value.upper()}

## Rationale
"""
        
        for i, rationale in enumerate(decision.rationale, 1):
            report += f"{i}. {rationale}\n"
        
        if decision.alternatives:
            report += "\n## Alternative Options\n"
            for alt in decision.alternatives:
                report += f"- {alt}\n"
        
        if decision.risks:
            report += "\n## Potential Risks\n"
            for risk in decision.risks:
                report += f"- {risk}\n"
        
        if decision.benefits:
            report += "\n## Expected Benefits\n"
            for benefit in decision.benefits:
                report += f"- {benefit}\n"
        
        report += "\n## Supporting Evidence\n"
        for evidence in decision.supporting_evidence:
            report += f"- Source: {evidence['source']}\n"
            report += f"  Evidence: {evidence['evidence']}\n"
        
        return report

def main():
    """Main function for clinical decision support."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clinical Decision Support System')
    parser.add_argument('--knowledge-path', type=str, default='./clinical_knowledge',
                       help='Path to clinical knowledge base')
    parser.add_argument('--patient-id', type=str, help='Patient ID')
    parser.add_argument('--medication', type=str, help='Medication name')
    parser.add_argument('--condition', type=str, help='Patient condition')
    parser.add_argument('--action', type=str, 
                       choices=['check_interaction', 'assess_risk', 'recommend', 'calculate_dosage'],
                       help='Action to perform')
    
    args = parser.parse_args()
    
    # Initialize decision support system
    cdss = ClinicalDecisionSupportSystem(args.knowledge_path)
    
    # Add sample patient profile
    sample_patient = PatientProfile(
        patient_id="patient_001",
        age=45,
        weight=70.0,
        height=175.0,
        gender="M",
        allergies=["penicillin"],
        current_medications=["lisinopril"],
        medical_conditions=["hypertension"],
        vital_signs={"bp_systolic": 130, "bp_diastolic": 85, "heart_rate": 72},
        lab_results={"creatinine": 1.0, "glucose": 100},
        contraindications=["pregnancy"],
        preferences={"language": "english"}
    )
    
    cdss.add_patient_profile(sample_patient)
    
    # Perform action
    if args.action == "check_interaction" and args.patient_id and args.medication:
        decision = cdss.check_medication_interactions(args.patient_id, args.medication)
        print(cdss.generate_decision_report(decision))
    
    elif args.action == "assess_risk" and args.patient_id and args.medication:
        decision = cdss.assess_medication_risk(args.patient_id, args.medication)
        print(cdss.generate_decision_report(decision))
    
    elif args.action == "recommend" and args.patient_id and args.condition:
        decision = cdss.recommend_medication(args.patient_id, args.condition)
        print(cdss.generate_decision_report(decision))
    
    elif args.action == "calculate_dosage" and args.patient_id and args.medication:
        decision = cdss.calculate_dosage(args.patient_id, args.medication)
        print(cdss.generate_decision_report(decision))
    
    else:
        print("Invalid arguments. Please provide patient_id, medication/condition, and action.")

if __name__ == "__main__":
    main()
