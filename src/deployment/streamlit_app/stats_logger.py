"""
Stats Logger - Enregistre toutes les requêtes pour analyse
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from config import STATS_LOG, CACHE_MISS_LOG

class StatsLogger:
    def __init__(self):
        # Créer les dossiers si nécessaire
        Path(STATS_LOG).parent.mkdir(parents=True, exist_ok=True)
        Path(CACHE_MISS_LOG).parent.mkdir(parents=True, exist_ok=True)

    def log_query(self, query_data: Dict[str, Any]) -> None:
        """
        Enregistre une requête dans stats.jsonl

        Args:
            query_data: {
                "query_id": str,
                "timestamp": str,
                "status": "hit" | "miss",
                "similarity_score": float,
                "confidence": float,
                "response_time_ms": float,
                "theme": str,
                "error_category": str,
                "difficulty": str,
                "deepseek_tokens": int,
                "cache_size": int
            }
        """
        # Ajouter timestamp si pas présent
        if 'timestamp' not in query_data:
            query_data['timestamp'] = datetime.now().isoformat()

        try:
            with open(STATS_LOG, 'a', encoding='utf-8') as f:
                f.write(json.dumps(query_data, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Error logging stats: {e}")

    def log_cache_miss(self, miss_data: Dict[str, Any]) -> None:
        """
        Enregistre un cache miss avec toutes les données (format dataset).

        Args:
            miss_data: {
                "theme": str,
                "difficulty": str,
                "tags": list,
                "error_category": str,
                "instructions": str,
                "code": str,
                "test_cases_scope": list,
                "failed_tests": list,
                "feedback": str,
                "query_id": str,
                "timestamp": str,
                "tokens_used": int
            }
        """
        # Ajouter timestamp
        if 'timestamp' not in miss_data:
            miss_data['timestamp'] = datetime.now().isoformat()

        try:
            with open(CACHE_MISS_LOG, 'a', encoding='utf-8') as f:
                f.write(json.dumps(miss_data, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Error logging cache miss: {e}")

    def read_stats(self, limit: int = None) -> list:
        """
        Lit les stats depuis le fichier.

        Args:
            limit: Nombre max de lignes à retourner (None = toutes)

        Returns:
            Liste de dicts
        """
        stats = []
        try:
            with open(STATS_LOG, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        stats.append(json.loads(line))

            if limit:
                return stats[-limit:]
            return stats

        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"Error reading stats: {e}")
            return []

    def read_cache_misses(self, limit: int = None) -> list:
        """
        Lit les cache misses depuis le fichier.

        Args:
            limit: Nombre max de lignes à retourner (None = toutes)

        Returns:
            Liste de dicts (format dataset)
        """
        misses = []
        try:
            with open(CACHE_MISS_LOG, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        misses.append(json.loads(line))

            if limit:
                return misses[-limit:]
            return misses

        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"Error reading cache misses: {e}")
            return []

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Retourne un résumé des statistiques.

        Returns:
            {
                "total_queries": int,
                "total_hits": int,
                "total_misses": int,
                "hit_rate": float,
                "avg_confidence": float,
                "total_deepseek_tokens": int,
                "avg_response_time_ms": float
            }
        """
        stats = self.read_stats()

        if not stats:
            return {
                "total_queries": 0,
                "total_hits": 0,
                "total_misses": 0,
                "hit_rate": 0.0,
                "avg_confidence": 0.0,
                "total_deepseek_tokens": 0,
                "avg_response_time_ms": 0.0
            }

        total_queries = len(stats)
        total_hits = sum(1 for s in stats if s.get('status') == 'hit')
        total_misses = total_queries - total_hits

        hit_rate = (total_hits / total_queries) * 100 if total_queries > 0 else 0.0

        confidences = [s.get('confidence', 0) for s in stats if s.get('confidence') is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        total_tokens = sum(s.get('deepseek_tokens', 0) for s in stats)

        response_times = [s.get('response_time_ms', 0) for s in stats if s.get('response_time_ms')]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0

        return {
            "total_queries": total_queries,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": round(hit_rate, 2),
            "avg_confidence": round(avg_confidence, 3),
            "total_deepseek_tokens": total_tokens,
            "avg_response_time_ms": round(avg_response_time, 2)
        }
