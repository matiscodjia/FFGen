"""
Cache Manager - Gère Hit/Miss et distillation
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import uuid
from datetime import datetime
from config import DISTANCE_THRESHOLD, TOP_K_RESULTS, CONFIDENCE_THRESHOLD_WARNING

class CacheManager:
    def __init__(self, chroma_collection, encoder_fn, threshold=None):
        """
        Args:
            chroma_collection: Collection ChromaDB
            encoder_fn: Fonction pour encoder du texte en embedding
            threshold: Custom similarity threshold (if None, uses config default)
        """
        self.collection = chroma_collection
        self.encoder_fn = encoder_fn
        self.threshold = threshold if threshold is not None else DISTANCE_THRESHOLD

    def calculate_confidence(self, distances: List[float]) -> float:
        """
        Calcule un score de confiance basé sur les distances.
        Distance plus faible = confiance plus haute.

        Returns:
            float entre 0 et 1
        """
        if not distances:
            return 0.0

        # Distance moyenne
        avg_distance = np.mean(distances)

        # Convertir distance en confiance (inverse et normalisation)
        # Distance de 0 = confiance 1.0
        # Distance de 0.5 = confiance 0.5
        # Distance de 1.0 = confiance 0.0
        confidence = max(0.0, 1.0 - avg_distance)

        return round(confidence, 3)

    def query_cache(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Logique d'exécution (Pipeline) :
        1. CHECK RAPIDE : Match exact de la chaîne de caractères (via Metadata).
           -> Si trouvé : Retour immédiat (Stop).
        
        2. RETRIEVAL : Recherche des 5 vecteurs les plus proches (Bi-Encoder).
        
        3. ANALYSE FINE : Sur ces 5 candidats, on vérifie :
           A. Est-ce qu'il y a un "Jumeau Sémantique" ? (Code quasi-identique > 0.95)
              -> Si oui : C'est un HIT forcé (Priorité sur le seuil).
           B. Est-ce que le meilleur candidat est sous le seuil de distance ?
              -> Si oui : C'est un HIT standard.
        
        4. DÉCISION : Si ni A ni B -> MISS.
        """
        
        # --- ÉTAPE 1 : CHECK RAPIDE (String Exact Match) ---
        try:
            # On vérifie si la chaîne de caractères brute existe déjà
            if len(code) < 5000: 
                exact_matches = self.collection.get(
                    where={"code": code},
                    limit=1
                )
                if exact_matches and len(exact_matches['ids']) > 0:
                    print("Cache: MATCH EXACT (String) trouvé !")
                    return {
                        "status": "perfect_match",
                        "results": [{
                            "feedback": exact_matches['documents'][0],
                            "code": code,
                            "distance": 0.0,
                            "rank": 1,
                            "metadata": exact_matches['metadatas'][0]
                        }],
                        "similarity_scores": [0.0],
                        "confidence": 1.0,
                        "needs_deepseek": False,
                        "needs_warning": False,
                        "query_id": str(uuid.uuid4()),
                        "query_embedding": [], 
                        "perfect_code_match": True
                    }
        except Exception as e:
            print(f"Warning exact match: {e}")

        # --- ÉTAPE 2 : RETRIEVAL (Recherche Vectorielle) ---
        # On a besoin des candidats pour faire les analyses suivantes
        
        query_embedding = self.encoder_fn(code)

        query_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K_RESULTS
        )

        distances = query_results['distances'][0] if query_results['distances'] else []
        documents = query_results['documents'][0] if query_results['documents'] else []
        metadatas = query_results['metadatas'][0] if query_results['metadatas'] else []

        # --- ÉTAPE 3 : ANALYSE FINE (Code Similarity Check) ---
        # On cherche un "Jumeau Sémantique" parmi les résultats retournés
        code_similarity = None
        perfect_code_match = False

        # On regarde uniquement le meilleur candidat (rank 1) pour la comparaison code-à-code
        if metadatas and metadatas[0].get('code'):
            ref_code = metadatas[0].get('code')
            if ref_code and ref_code != 'N/A':
                ref_code_embedding = self.encoder_fn(ref_code)
                # Produit scalaire
                code_similarity = float(np.dot(query_embedding, ref_code_embedding))

                # Si > 0.95, c'est le même code écrit différemment (ex: espaces, commentaires)
                if code_similarity > 0.95:
                    perfect_code_match = True

        # --- ÉTAPE 4 : DÉCISION HIT / MISS ---
        
        # Condition A : Jumeau Sémantique (Le code est quasi identique)
        # Condition B : Proximité Vectorielle Standard (Le sens est proche, sous le seuil)
        
        is_hit = False
        hit_type = "miss"

        if perfect_code_match:
            is_hit = True
            hit_type = "perfect_match" # Priorité haute
        elif distances and distances[0] < self.threshold:
            is_hit = True
            hit_type = "hit" # Priorité standard

        # --- CONSTRUCTION DE LA RÉPONSE ---
        
        # Préparation des résultats formatés (utilisé dans les deux cas)
        formatted_results = []
        for i, (feedback, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            formatted_results.append({
                "rank": i + 1,
                "feedback": feedback,
                "code": metadata.get('code', 'N/A'),
                "distance": round(distance, 4),
                "metadata": metadata
            })

        if is_hit:
            # Calcul confiance
            confidence = self.calculate_confidence(distances)
            if perfect_code_match:
                confidence = 1.0 # Boost max car on est sûr du code

            return {
                "status": hit_type,
                "results": formatted_results,
                "similarity_scores": [round(d, 4) for d in distances],
                "confidence": confidence,
                "needs_deepseek": False,
                # Warning uniquement si c'est un hit "mou" (vecteur lointain) ET pas un match de code
                "needs_warning": False if perfect_code_match else (confidence < CONFIDENCE_THRESHOLD_WARNING),
                "query_embedding": query_embedding,
                "query_id": str(uuid.uuid4()),
                "code_similarity": round(code_similarity, 4) if code_similarity is not None else None,
                "perfect_code_match": perfect_code_match
            }

        else:
            # MISS
            return {
                "status": "miss",
                "results": formatted_results, # On renvoie quand même les proches pour info
                "similarity_scores": [round(d, 4) for d in distances] if distances else [],
                "confidence": 0.0,
                "needs_deepseek": True,
                "needs_warning": False,
                "query_embedding": query_embedding,
                "query_id": str(uuid.uuid4()),
                "closest_distance": round(distances[0], 4) if distances else 1.0
            }
    def add_to_cache(self, code: str, feedback: str, metadata: Dict[str, Any], embedding: List[float]) -> bool:
        """
        Ajoute une nouvelle entrée au cache (distillation online).

        Args:
            code: Code source
            feedback: Feedback généré
            metadata: Métadonnées complètes (theme, difficulty, etc.)
            embedding: Embedding du feedback

        Returns:
            bool: True si succès
        """
        try:
            doc_id = f"miss_{uuid.uuid4()}"

            # Préparer metadata pour ChromaDB (seulement le code car limitation)
            chroma_metadata = {
                "code": code,
                "timestamp": datetime.now().isoformat(),
                "source": "cache_miss"
            }

            self.collection.add(
                embeddings=[embedding],
                documents=[feedback],
                metadatas=[chroma_metadata],
                ids=[doc_id]
            )

            return True

        except Exception as e:
            print(f"Error adding to cache: {e}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne des stats sur le cache"""
        try:
            total_docs = self.collection.count()

            return {
                "total_documents": total_docs,
                "similarity_threshold": DISTANCE_THRESHOLD,
                "top_k": TOP_K_RESULTS
            }
        except Exception as e:
            return {"error": str(e)}
