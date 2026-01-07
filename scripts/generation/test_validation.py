#!/usr/bin/env python3
"""
Script de test pour valider le système de génération
Test sur 5 samples avant de lancer la grosse prod
"""

import os
import sys
import json

# Import des fonctions de validation
sys.path.insert(0, os.path.dirname(__file__))
from generate_dataset_validated import validate_compilation, validate_semantic_bug, generate_synthetic_entry

def test_validation_functions():
    """Test unitaire des fonctions de validation"""
    print("=" * 70)
    print("🧪 TEST 1: Validation de compilation")
    print("=" * 70)

    # Test 1a: Code correct qui compile
    good_code = """
#include <stdio.h>

int main() {
    printf("Hello World\\n");
    return 0;
}
"""
    success, msg = validate_compilation(good_code)
    print(f"✅ Code correct: {success} - {msg}")
    assert success, "Le code correct devrait compiler"

    # Test 1b: Code avec erreur de syntaxe
    bad_syntax = """
#include <stdio.h>

int main() {
    printf("Missing semicolon")
    return 0;
}
"""
    success, msg = validate_compilation(bad_syntax)
    print(f"❌ Erreur syntaxe: {success} - {msg[:80]}")
    assert not success, "Le code avec erreur syntaxe ne devrait PAS compiler"

    # Test 1c: Code avec bug sémantique (compile mais incorrect)
    semantic_bug = """
#include <stdio.h>

int is_prime(int n) {
    if (n <= 1) {
        return 1;  // Bug: devrait retourner 0
    }
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}

int main() {
    printf("%d\\n", is_prime(1));
    return 0;
}
"""
    success, msg = validate_compilation(semantic_bug)
    print(f"✅ Bug sémantique (compile): {success} - {msg}")
    assert success, "Le code avec bug sémantique devrait compiler"

    print()
    print("=" * 70)
    print("🧪 TEST 2: Validation sémantique du JSON")
    print("=" * 70)

    # Test 2a: JSON valide
    valid_json = {
        "theme": "Check for Prime Number",
        "difficulty": "beginner",
        "tags": ["math", "logic"],
        "error_category": "Incorrect Base Case Return Value",
        "instructions": "Check if number is prime",
        "code": semantic_bug,
        "test_cases_scope": ["Input: 1", "Input: 2"],
        "failed_tests": ["Input: 1 (returns 1, expected 0)"],
        "feedback": "Review the definition of prime numbers"
    }
    success, msg = validate_semantic_bug(valid_json)
    print(f"✅ JSON valide: {success} - {msg}")
    assert success, "Le JSON valide devrait passer"

    # Test 2b: JSON avec tests qui passent (phantom bug)
    phantom_bug = valid_json.copy()
    phantom_bug["failed_tests"] = ["Test 1: outputs '0' (correct)", "Test 2: outputs '1' (correct)"]
    success, msg = validate_semantic_bug(phantom_bug)
    print(f"❌ Phantom bug détecté: {success} - {msg}")
    assert not success, "Le phantom bug devrait être rejeté"

    # Test 2c: JSON incomplet
    incomplete = {"theme": "Test", "code": "int main() {}"}
    success, msg = validate_semantic_bug(incomplete)
    print(f"❌ JSON incomplet: {success} - {msg}")
    assert not success, "Le JSON incomplet devrait être rejeté"

    print()
    print("=" * 70)
    print("✅ TOUS LES TESTS UNITAIRES PASSENT")
    print("=" * 70)
    print()

def test_full_generation():
    """Test de génération complète (1 sample)"""
    print("=" * 70)
    print("🧪 TEST 3: Génération complète d'un sample")
    print("=" * 70)
    print("⚠️  Ce test va consommer des tokens API...")
    print()

    exercise = "Check for Prime Number"
    result = generate_synthetic_entry(exercise)

    if result:
        print()
        print("✅ SAMPLE GÉNÉRÉ AVEC SUCCÈS")
        print("-" * 70)
        print(f"Theme: {result['theme']}")
        print(f"Difficulty: {result['difficulty']}")
        print(f"Tags: {', '.join(result['tags'])}")
        print(f"Error: {result['error_category']}")
        print(f"Failed tests: {len(result['failed_tests'])} tests")
        print()
        print("Code preview:")
        print(result['code'][:300] + "...")
        print("-" * 70)
        return True
    else:
        print()
        print("❌ ÉCHEC DE GÉNÉRATION (peut-être rate limit ou mauvaise config API)")
        return False

if __name__ == "__main__":
    print("🚀 DÉMARRAGE DES TESTS DE VALIDATION")
    print()

    # Tests unitaires (pas de consommation de tokens)
    try:
        test_validation_functions()
    except AssertionError as e:
        print(f"💥 ÉCHEC DES TESTS UNITAIRES: {e}")
        sys.exit(1)

    # Test de génération (consomme des tokens)
    if os.environ.get('DEEPSEEK_API_KEY'):
        response = input("\n🔑 API key détectée. Lancer le test de génération (consomme tokens) ? [y/N]: ")
        if response.lower() == 'y':
            success = test_full_generation()
            if success:
                print()
                print("=" * 70)
                print("✅ SYSTÈME VALIDÉ - Prêt pour la production !")
                print("=" * 70)
                print()
                print("Pour lancer la génération complète:")
                print("  python scripts/generate_dataset_validated.py")
            else:
                print()
                print("⚠️  Vérifiez votre configuration API")
                sys.exit(1)
        else:
            print("⏭️  Test de génération skippé")
    else:
        print("⚠️  Aucune API key détectée. Test de génération skippé.")
        print("   Définissez DEEPSEEK_API_KEY pour tester la génération")

    print()
    print("🎉 TESTS TERMINÉS")
