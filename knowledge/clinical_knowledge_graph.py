#!/usr/bin/env python3
"""
Clinical Knowledge Graph System
Knowledge representation and reasoning for clinical robotics applications.

This module implements:
- Clinical knowledge graph construction
- Medical ontology integration
- Drug-disease interaction modeling
- Clinical guideline representation
- Knowledge graph querying and reasoning
- Clinical decision support
- Knowledge extraction from clinical texts
- Graph-based recommendation systems

Author: Idriss Djiofack Teledjieu
Clinical Robot Adaptation Project
HIRO Laboratory, University of Colorado Boulder
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import warnings

# Knowledge graph libraries
import networkx as nx
from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS, OWL
from rdflib.namespace import XSD

# NLP for knowledge extraction
import spacy
from transformers import AutoTokenizer, AutoModel
import torch

# Database
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clinical_knowledge_graph.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class EntityType(Enum):
    """Types of entities in clinical knowledge graph."""
    MEDICATION = "medication"
    DISEASE = "disease"
    SYMPTOM = "symptom"
    PROCEDURE = "procedure"
    DEVICE = "device"
    PATIENT = "patient"
    CLINICIAN = "clinician"
    HOSPITAL = "hospital"
    DEPARTMENT = "department"
    LAB_TEST = "lab_test"
    TREATMENT = "treatment"
    CONTRAINDICATION = "contraindication"
    SIDE_EFFECT = "side_effect"

class RelationType(Enum):
    """Types of relations in clinical knowledge graph."""
    TREATS = "treats"
    CAUSES = "causes"
    SYMPTOM_OF = "symptom_of"
    DIAGNOSED_WITH = "diagnosed_with"
    PRESCRIBES = "prescribes"
    ADMINISTERS = "administers"
    CONTRAINDICATED_FOR = "contraindicated_for"
    HAS_SIDE_EFFECT = "has_side_effect"
    INTERACTS_WITH = "interacts_with"
    REQUIRES = "requires"
    PERFORMS = "performs"
    WORKS_AT = "works_at"
    SPECIALIZES_IN = "specializes_in"
    LOCATED_IN = "located_in"
    TEST_FOR = "test_for"

@dataclass
class Entity:
    """Entity in clinical knowledge graph."""
    entity_id: str
    entity_type: EntityType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    synonyms: List[str] = field(default_factory=list)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entity_id': self.entity_id,
            'entity_type': self.entity_type.value,
            'name': self.name,
            'properties': self.properties,
            'synonyms': self.synonyms,
            'description': self.description
        }

@dataclass
class Relation:
    """Relation between entities in clinical knowledge graph."""
    relation_id: str
    relation_type: RelationType
    source_entity: str
    target_entity: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'relation_id': self.relation_id,
            'relation_type': self.relation_type.value,
            'source_entity': self.source_entity,
            'target_entity': self.target_entity,
            'properties': self.properties,
            'confidence': self.confidence,
            'evidence': self.evidence
        }

@dataclass
class KnowledgeGraphConfig:
    """Configuration for clinical knowledge graph."""
    
    # Graph construction
    enable_ontology_import: bool = True
    ontology_sources: List[str] = field(default_factory=lambda: ["SNOMED_CT", "RxNorm", "ICD-10"])
    
    # Knowledge extraction
    enable_nlp_extraction: bool = True
    nlp_model: str = "en_core_sci_md"  # Medical NLP model
    transformer_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased"
    
    # Graph storage
    storage_backend: str = "networkx"  # networkx, neo4j, rdf
    database_url: str = "sqlite:///clinical_knowledge.db"
    
    # Reasoning
    enable_reasoning: bool = True
    reasoning_engine: str = "rule_based"  # rule_based, probabilistic, neural
    
    # Validation
    enable_validation: bool = True
    confidence_threshold: float = 0.7
    
    # Export
    export_formats: List[str] = field(default_factory=lambda: ["json", "rdf", "graphml"])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'enable_ontology_import': self.enable_ontology_import,
            'ontology_sources': self.ontology_sources,
            'enable_nlp_extraction': self.enable_nlp_extraction,
            'nlp_model': self.nlp_model,
            'transformer_model': self.transformer_model,
            'storage_backend': self.storage_backend,
            'database_url': self.database_url,
            'enable_reasoning': self.enable_reasoning,
            'reasoning_engine': self.reasoning_engine,
            'enable_validation': self.enable_validation,
            'confidence_threshold': self.confidence_threshold,
            'export_formats': self.export_formats
        }

# Database models
Base = declarative_base()

class KnowledgeEntity(Base):
    """Entity database model."""
    __tablename__ = "knowledge_entities"
    
    id = Column(Integer, primary_key=True, index=True)
    entity_id = Column(String(100), unique=True, nullable=False, index=True)
    entity_type = Column(String(50), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    properties = Column(Text, nullable=True)  # JSON string
    synonyms = Column(Text, nullable=True)  # JSON string
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

class KnowledgeRelation(Base):
    """Relation database model."""
    __tablename__ = "knowledge_relations"
    
    id = Column(Integer, primary_key=True, index=True)
    relation_id = Column(String(100), unique=True, nullable=False, index=True)
    relation_type = Column(String(50), nullable=False, index=True)
    source_entity = Column(String(100), nullable=False, index=True)
    target_entity = Column(String(100), nullable=False, index=True)
    properties = Column(Text, nullable=True)  # JSON string
    confidence = Column(Float, nullable=False)
    evidence = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=datetime.now, nullable=False)

class ClinicalKnowledgeGraph:
    """Main clinical knowledge graph system."""
    
    def __init__(self, config: KnowledgeGraphConfig):
        self.config = config
        
        # Initialize graph
        self.graph = nx.DiGraph()
        
        # Initialize NLP components
        self.nlp = None
        self.transformer_tokenizer = None
        self.transformer_model = None
        
        if config.enable_nlp_extraction:
            self._initialize_nlp()
        
        # Initialize database
        self.engine = create_engine(config.database_url)
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Load initial knowledge
        self._load_clinical_knowledge()
        
        logger.info("Clinical Knowledge Graph initialized")
    
    def _initialize_nlp(self):
        """Initialize NLP components for knowledge extraction."""
        try:
            # Load spaCy medical model
            self.nlp = spacy.load(self.config.nlp_model)
            logger.info(f"Loaded spaCy model: {self.config.nlp_model}")
        except OSError:
            logger.warning(f"Could not load {self.config.nlp_model}, using en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Load transformer model
        try:
            self.transformer_tokenizer = AutoTokenizer.from_pretrained(self.config.transformer_model)
            self.transformer_model = AutoModel.from_pretrained(self.config.transformer_model)
            logger.info(f"Loaded transformer model: {self.config.transformer_model}")
        except Exception as e:
            logger.warning(f"Could not load transformer model: {e}")
    
    def _load_clinical_knowledge(self):
        """Load initial clinical knowledge."""
        # Add medications
        medications = [
            ("aspirin", "NSAID", ["acetylsalicylic acid", "ASA"], "Pain reliever and anti-inflammatory"),
            ("ibuprofen", "NSAID", ["advil", "motrin"], "NSAID for pain and inflammation"),
            ("acetaminophen", "analgesic", ["paracetamol", "tylenol"], "Pain reliever and fever reducer"),
            ("metformin", "antidiabetic", ["glucophage"], "First-line treatment for type 2 diabetes"),
            ("lisinopril", "ace_inhibitor", ["prinivil", "zestril"], "Treatment for hypertension and heart failure"),
            ("amoxicillin", "antibiotic", ["amoxil"], "Penicillin antibiotic for bacterial infections"),
            ("warfarin", "anticoagulant", ["coumadin"], "Blood thinner for preventing clots"),
            ("insulin", "hormone", ["humulin", "novolin"], "Hormone for diabetes management"),
            ("morphine", "opioid", ["ms contin", "avinza"], "Strong pain reliever")
        ]
        
        for name, med_type, synonyms, description in medications:
            entity = Entity(
                entity_id=f"med_{name.lower()}",
                entity_type=EntityType.MEDICATION,
                name=name,
                properties={"medication_type": med_type},
                synonyms=synonyms,
                description=description
            )
            self.add_entity(entity)
        
        # Add diseases
        diseases = [
            ("hypertension", "cardiovascular", ["high blood pressure"], "Chronic high blood pressure"),
            ("diabetes_type2", "endocrine", ["type 2 diabetes", "t2dm"], "Metabolic disorder affecting blood sugar"),
            ("pneumonia", "respiratory", ["lung infection"], "Infection of the lungs"),
            ("myocardial_infarction", "cardiovascular", ["heart attack", "mi"], "Heart muscle damage"),
            ("stroke", "neurological", ["cerebrovascular accident"], "Brain blood flow interruption"),
            ("arthritis", "musculoskeletal", ["joint inflammation"], "Joint inflammation and pain"),
            ("asthma", "respiratory", ["bronchial asthma"], "Chronic airway disease"),
            ("depression", "mental", ["major depressive disorder"], "Mood disorder"),
            ("cancer", "neoplastic", ["malignancy", "tumor"], "Uncontrolled cell growth")
        ]
        
        for name, disease_type, synonyms, description in diseases:
            entity = Entity(
                entity_id=f"disease_{name.lower()}",
                entity_type=EntityType.DISEASE,
                name=name,
                properties={"disease_type": disease_type},
                synonyms=synonyms,
                description=description
            )
            self.add_entity(entity)
        
        # Add relations
        self._add_medication_disease_relations()
        self._add_medication_contraindications()
        self._add_disease_symptoms()
        
        logger.info("Initial clinical knowledge loaded")
    
    def _add_medication_disease_relations(self):
        """Add medication-disease treatment relations."""
        treatment_relations = [
            ("aspirin", "hypertension", RelationType.TREATS),
            ("aspirin", "myocardial_infarction", RelationType.TREATS),
            ("metformin", "diabetes_type2", RelationType.TREATS),
            ("lisinopril", "hypertension", RelationType.TREATS),
            ("lisinopril", "myocardial_infarction", RelationType.TREATS),
            ("insulin", "diabetes_type2", RelationType.TREATS),
            ("warfarin", "stroke", RelationType.TREATS),
            ("amoxicillin", "pneumonia", RelationType.TREATS)
        ]
        
        for med, disease, relation_type in treatment_relations:
            relation = Relation(
                relation_id=f"{med}_{disease}_{relation_type.value}",
                relation_type=relation_type,
                source_entity=f"med_{med.lower()}",
                target_entity=f"disease_{disease.lower()}",
                confidence=0.9
            )
            self.add_relation(relation)
    
    def _add_medication_contraindications(self):
        """Add medication contraindication relations."""
        contraindications = [
            ("aspirin", "bleeding_disorder", RelationType.CONTRAINDICATED_FOR),
            ("ibuprofen", "kidney_disease", RelationType.CONTRAINDICATED_FOR),
            ("metformin", "liver_disease", RelationType.CONTRAINDICATED_FOR),
            ("lisinopril", "pregnancy", RelationType.CONTRAINDICATED_FOR),
            ("warfarin", "bleeding_disorder", RelationType.CONTRAINDICATED_FOR)
        ]
        
        for med, condition, relation_type in contraindications:
            # Add condition as entity if not exists
            condition_entity = Entity(
                entity_id=f"condition_{condition.lower()}",
                entity_type=EntityType.CONTRAINDICATION,
                name=condition,
                description=f"Medical condition: {condition}"
            )
            self.add_entity(condition_entity)
            
            relation = Relation(
                relation_id=f"{med}_{condition}_{relation_type.value}",
                relation_type=relation_type,
                source_entity=f"med_{med.lower()}",
                target_entity=f"condition_{condition.lower()}",
                confidence=0.95
            )
            self.add_relation(relation)
    
    def _add_disease_symptoms(self):
        """Add disease-symptom relations."""
        disease_symptoms = [
            ("hypertension", "headache", RelationType.SYMPTOM_OF),
            ("hypertension", "dizziness", RelationType.SYMPTOM_OF),
            ("hypertension", "blurred_vision", RelationType.SYMPTOM_OF),
            ("diabetes_type2", "increased_thirst", RelationType.SYMPTOM_OF),
            ("diabetes_type2", "frequent_urination", RelationType.SYMPTOM_OF),
            ("diabetes_type2", "fatigue", RelationType.SYMPTOM_OF),
            ("pneumonia", "cough", RelationType.SYMPTOM_OF),
            ("pneumonia", "fever", RelationType.SYMPTOM_OF),
            ("pneumonia", "shortness_of_breath", RelationType.SYMPTOM_OF),
            ("myocardial_infarction", "chest_pain", RelationType.SYMPTOM_OF),
            ("myocardial_infarction", "shortness_of_breath", RelationType.SYMPTOM_OF),
            ("stroke", "numbness", RelationType.SYMPTOM_OF),
            ("stroke", "confusion", RelationType.SYMPTOM_OF),
            ("stroke", "trouble_speaking", RelationType.SYMPTOM_OF)
        ]
        
        for disease, symptom, relation_type in disease_symptoms:
            # Add symptom as entity if not exists
            symptom_entity = Entity(
                entity_id=f"symptom_{symptom.lower()}",
                entity_type=EntityType.SYMPTOM,
                name=symptom,
                description=f"Symptom: {symptom}"
            )
            self.add_entity(symptom_entity)
            
            relation = Relation(
                relation_id=f"{disease}_{symptom}_{relation_type.value}",
                relation_type=relation_type,
                source_entity=f"symptom_{symptom.lower()}",
                target_entity=f"disease_{disease.lower()}",
                confidence=0.85
            )
            self.add_relation(relation)
    
    def add_entity(self, entity: Entity):
        """Add entity to knowledge graph."""
        # Add to NetworkX graph
        self.graph.add_node(
            entity.entity_id,
            entity_type=entity.entity_type.value,
            name=entity.name,
            properties=entity.properties,
            synonyms=entity.synonyms,
            description=entity.description
        )
        
        # Add to database
        db = self.SessionLocal()
        try:
            db_entity = KnowledgeEntity(
                entity_id=entity.entity_id,
                entity_type=entity.entity_type.value,
                name=entity.name,
                properties=json.dumps(entity.properties),
                synonyms=json.dumps(entity.synonyms),
                description=entity.description
            )
            db.add(db_entity)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to add entity to database: {e}")
        finally:
            db.close()
        
        logger.info(f"Entity added: {entity.name}")
    
    def add_relation(self, relation: Relation):
        """Add relation to knowledge graph."""
        # Add to NetworkX graph
        self.graph.add_edge(
            relation.source_entity,
            relation.target_entity,
            relation_type=relation.relation_type.value,
            confidence=relation.confidence,
            properties=relation.properties,
            evidence=relation.evidence
        )
        
        # Add to database
        db = self.SessionLocal()
        try:
            db_relation = KnowledgeRelation(
                relation_id=relation.relation_id,
                relation_type=relation.relation_type.value,
                source_entity=relation.source_entity,
                target_entity=relation.target_entity,
                properties=json.dumps(relation.properties),
                confidence=relation.confidence,
                evidence=json.dumps(relation.evidence)
            )
            db.add(db_relation)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to add relation to database: {e}")
        finally:
            db.close()
        
        logger.info(f"Relation added: {relation.relation_type.value}")
    
    def query_entity(self, entity_name: str, entity_type: EntityType = None) -> List[Entity]:
        """Query entities by name and type."""
        results = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            name_match = entity_name.lower() in node_data['name'].lower()
            type_match = entity_type is None or node_data['entity_type'] == entity_type.value
            
            if name_match and type_match:
                entity = Entity(
                    entity_id=node_id,
                    entity_type=EntityType(node_data['entity_type']),
                    name=node_data['name'],
                    properties=node_data.get('properties', {}),
                    synonyms=node_data.get('synonyms', []),
                    description=node_data.get('description', '')
                )
                results.append(entity)
        
        return results
    
    def query_relations(self, source_entity: str = None, target_entity: str = None,
                       relation_type: RelationType = None) -> List[Relation]:
        """Query relations by source, target, and type."""
        results = []
        
        for source, target, edge_data in self.graph.edges(data=True):
            source_match = source_entity is None or source == source_entity
            target_match = target_entity is None or target == target_entity
            type_match = relation_type is None or edge_data['relation_type'] == relation_type.value
            
            if source_match and target_match and type_match:
                relation = Relation(
                    relation_id=f"{source}_{target}_{edge_data['relation_type']}",
                    relation_type=RelationType(edge_data['relation_type']),
                    source_entity=source,
                    target_entity=target,
                    properties=edge_data.get('properties', {}),
                    confidence=edge_data.get('confidence', 1.0),
                    evidence=edge_data.get('evidence', [])
                )
                results.append(relation)
        
        return results
    
    def find_treatments(self, disease: str) -> List[Entity]:
        """Find treatments for a given disease."""
        disease_entity = self.query_entity(disease, EntityType.DISEASE)
        
        if not disease_entity:
            return []
        
        disease_id = disease_entity[0].entity_id
        
        # Find medications that treat this disease
        relations = self.query_relations(target_entity=disease_id, relation_type=RelationType.TREATS)
        
        treatments = []
        for relation in relations:
            med_entity = self.graph.nodes[relation.source_entity]
            entity = Entity(
                entity_id=relation.source_entity,
                entity_type=EntityType(med_entity['entity_type']),
                name=med_entity['name'],
                properties=med_entity.get('properties', {}),
                synonyms=med_entity.get('synonyms', []),
                description=med_entity.get('description', '')
            )
            treatments.append(entity)
        
        return treatments
    
    def find_contraindications(self, medication: str, patient_conditions: List[str]) -> List[Entity]:
        """Find contraindications for medication given patient conditions."""
        med_entity = self.query_entity(medication, EntityType.MEDICATION)
        
        if not med_entity:
            return []
        
        med_id = med_entity[0].entity_id
        
        # Find contraindications
        relations = self.query_relations(source_entity=med_id, relation_type=RelationType.CONTRAINDICATED_FOR)
        
        contraindications = []
        for relation in relations:
            condition_name = self.graph.nodes[relation.target_entity]['name']
            
            # Check if patient has this condition
            if any(condition.lower() in patient_condition.lower() for patient_condition in patient_conditions):
                entity = Entity(
                    entity_id=relation.target_entity,
                    entity_type=EntityType(self.graph.nodes[relation.target_entity]['entity_type']),
                    name=condition_name,
                    properties=self.graph.nodes[relation.target_entity].get('properties', {}),
                    synonyms=self.graph.nodes[relation.target_entity].get('synonyms', []),
                    description=self.graph.nodes[relation.target_entity].get('description', '')
                )
                contraindications.append(entity)
        
        return contraindications
    
    def find_symptoms(self, disease: str) -> List[Entity]:
        """Find symptoms for a given disease."""
        disease_entity = self.query_entity(disease, EntityType.DISEASE)
        
        if not disease_entity:
            return []
        
        disease_id = disease_entity[0].entity_id
        
        # Find symptoms
        relations = self.query_relations(target_entity=disease_id, relation_type=RelationType.SYMPTOM_OF)
        
        symptoms = []
        for relation in relations:
            symptom_entity = self.graph.nodes[relation.source_entity]
            entity = Entity(
                entity_id=relation.source_entity,
                entity_type=EntityType(symptom_entity['entity_type']),
                name=symptom_entity['name'],
                properties=symptom_entity.get('properties', {}),
                synonyms=symptom_entity.get('synonyms', []),
                description=symptom_entity.get('description', '')
            )
            symptoms.append(entity)
        
        return symptoms
    
    def find_diseases_from_symptoms(self, symptoms: List[str]) -> List[Entity]:
        """Find diseases based on symptoms."""
        diseases = defaultdict(int)
        
        for symptom in symptoms:
            symptom_entity = self.query_entity(symptom, EntityType.SYMPTOM)
            
            if symptom_entity:
                symptom_id = symptom_entity[0].entity_id
                
                # Find diseases this symptom is related to
                relations = self.query_relations(source_entity=symptom_id, relation_type=RelationType.SYMPTOM_OF)
                
                for relation in relations:
                    disease_name = self.graph.nodes[relation.target_entity]['name']
                    diseases[disease_name] += 1
        
        # Sort by number of matching symptoms
        sorted_diseases = sorted(diseases.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for disease_name, count in sorted_diseases:
            disease_entity = self.query_entity(disease_name, EntityType.DISEASE)
            if disease_entity:
                results.append(disease_entity[0])
        
        return results
    
    def extract_knowledge_from_text(self, text: str) -> List[Tuple[Entity, Entity, Relation]]:
        """Extract knowledge from clinical text using NLP."""
        if self.nlp is None:
            return []
        
        extracted_relations = []
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract entities (simplified)
        for ent in doc.ents:
            if ent.label_ == "DISEASE":
                entity = Entity(
                    entity_id=f"extracted_disease_{ent.text.lower()}",
                    entity_type=EntityType.DISEASE,
                    name=ent.text,
                    description=f"Extracted disease: {ent.text}"
                )
                self.add_entity(entity)
        
            elif ent.label_ == "CHEMICAL":
                entity = Entity(
                    entity_id=f"extracted_med_{ent.text.lower()}",
                    entity_type=EntityType.MEDICATION,
                    name=ent.text,
                    description=f"Extracted medication: {ent.text}"
                )
                self.add_entity(entity)
        
        # Extract relations using dependency parsing (simplified)
        for token in doc:
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                # Subject-verb relation might indicate treatment
                pass
        
        return extracted_relations
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        return {
            'num_entities': self.graph.number_of_nodes(),
            'num_relations': self.graph.number_of_edges(),
            'entity_types': self._count_entity_types(),
            'relation_types': self._count_relation_types(),
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0
        }
    
    def _count_entity_types(self) -> Dict[str, int]:
        """Count entities by type."""
        type_counts = defaultdict(int)
        
        for node_data in self.graph.nodes(data=True):
            type_counts[node_data['entity_type']] += 1
        
        return dict(type_counts)
    
    def _count_relation_types(self) -> Dict[str, int]:
        """Count relations by type."""
        type_counts = defaultdict(int)
        
        for edge_data in self.graph.edges(data=True):
            type_counts[edge_data['relation_type']] += 1
        
        return dict(type_counts)
    
    def export_graph(self, output_path: str, format: str = "json"):
        """Export knowledge graph to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            # Export as JSON
            graph_data = {
                'entities': [self.graph.nodes[n] for n in self.graph.nodes()],
                'relations': [
                    {
                        'source': s,
                        'target': t,
                        **d
                    }
                    for s, t, d in self.graph.edges(data=True)
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
        
        elif format == "graphml":
            # Export as GraphML
            nx.write_graphml(self.graph, output_path)
        
        elif format == "rdf":
            # Export as RDF
            g = Graph()
            ex = Namespace("http://example.org/clinical/")
            
            for node_id, node_data in self.graph.nodes(data=True):
                uri = URIRef(ex[node_id])
                g.add((uri, RDF.type, OWL.NamedIndividual))
                g.add((uri, ex.name, Literal(node_data['name'])))
            
            for source, target, edge_data in self.graph.edges(data=True):
                g.add((URIRef(ex[source]), URIRef(ex[edge_data['relation_type']]), URIRef(ex[target])))
            
            g.serialize(output_path, format='xml')
        
        logger.info(f"Graph exported to: {output_path}")
    
    def visualize_graph(self, output_path: str = None):
        """Visualize knowledge graph."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        
        # Create layout
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Draw nodes
        node_colors = []
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node]['entity_type']
            if node_type == 'medication':
                node_colors.append('#ff6b6b')
            elif node_type == 'disease':
                node_colors.append('#4ecdc4')
            elif node_type == 'symptom':
                node_colors.append('#ffd93d')
            else:
                node_colors.append('#6bcb77')
        
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=500, alpha=0.7)
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3, width=1)
        nx.draw_networkx_labels(self.graph, pos, font_size=8, font_weight='bold')
        
        plt.title("Clinical Knowledge Graph")
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graph visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()

def main():
    """Main function for clinical knowledge graph."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clinical Knowledge Graph System')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--action', type=str, default='query',
                       choices=['query', 'extract', 'export', 'visualize', 'stats'],
                       help='Action to perform')
    parser.add_argument('--entity', type=str, help='Entity name to query')
    parser.add_argument('--type', type=str, help='Entity type')
    parser.add_argument('--symptoms', nargs='+', help='Symptoms for disease diagnosis')
    parser.add_argument('--output', type=str, help='Output path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = KnowledgeGraphConfig()
    
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Create knowledge graph
    kg = ClinicalKnowledgeGraph(config)
    
    # Perform action
    if args.action == "query":
        if args.entity:
            if args.type:
                entity_type = EntityType(args.type)
            else:
                entity_type = None
            
            entities = kg.query_entity(args.entity, entity_type)
            
            print(f"Query results for '{args.entity}':")
            for entity in entities:
                print(f"  - {entity.name} ({entity.entity_type.value})")
                print(f"    Description: {entity.description}")
                print(f"    Properties: {entity.properties}")
        else:
            print("Please provide an entity name to query")
    
    elif args.action == "extract":
        text = input("Enter clinical text to extract knowledge from: ")
        relations = kg.extract_knowledge_from_text(text)
        print(f"Extracted {len(relations)} relations")
    
    elif args.action == "export":
        if args.output:
            kg.export_graph(args.output, format="json")
            print(f"Graph exported to: {args.output}")
        else:
            kg.export_graph("clinical_knowledge_graph.json", format="json")
            print("Graph exported to: clinical_knowledge_graph.json")
    
    elif args.action == "visualize":
        kg.visualize_graph(args.output)
    
    elif args.action == "stats":
        stats = kg.get_graph_statistics()
        print("Knowledge Graph Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
