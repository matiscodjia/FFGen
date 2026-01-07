import os
import json
import time
import random
import threading
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from datetime import datetime

# ==============================================================================
# 1. CONFIGURATION & SÉCURITÉ
# ==============================================================================
api_key = os.environ.get('DEEPSEEK_API_KEY')
if not api_key:
    raise ValueError("CRITICAL: Environment variable DEEPSEEK_API_KEY is missing.")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

OUTPUT_FILE = "dataset_c_piscine_semantic_validated.jsonl"
METRICS_FILE = "generation_metrics.jsonl"
SUMMARY_FILE = "generation_summary.json"
TOTAL_SAMPLES_TARGET = 15000
MAX_WORKERS = 5
MAX_RETRIES = 2
FILE_LOCK = threading.Lock()
METRICS_LOCK = threading.Lock()

# Statistiques globales (thread-safe)
GLOBAL_STATS = {
    "total_attempts": 0,
    "total_successes": 0,
    "total_failures": 0,
    "total_tokens_prompt": 0,
    "total_tokens_completion": 0,
    "total_tokens_total": 0,
    "total_time_seconds": 0,
    "failure_reasons": {},
    "start_time": None,
    "end_time": None
}

# ==============================================================================
# 2. TAXONOMIE DES ERREURS (SÉMANTIQUES UNIQUEMENT)
# ==============================================================================
ERROR_TAXONOMY = [
    # --- MEMORY MANAGEMENT ---
    "Memory Leak (Missing free)",
    "Double Free",
    "Use After Free",
    "Dangling Pointer (Returning stack address)",
    "Allocation Size Error (sizeof type mismatch)",
    "Missing Null Terminator Space (malloc(strlen) instead of +1)",
    "Null Pointer Dereference (Missing NULL check)",

    # --- POINTERS & ARRAYS ---
    "Buffer Overflow (Write past end)",
    "Buffer Underflow (Read before start)",
    "Pointer Arithmetic Logic Error (*ptr++ vs (*ptr)++)",
    "Array decay misunderstanding (sizeof on array parameter)",

    # --- LOGIC & CONTROL FLOW ---
    "Off-by-one Error (Loop boundary <= vs <)",
    "Off-by-one Error (0-indexed vs 1-indexed)",
    "Infinite Loop (Bad update condition)",
    "Unreachable Code (Return in loop)",
    "Shadowing Variable (Local variable hides parameter)",
    "Assignment in Condition (if (x=y))",
    "Integer Overflow (Unchecked addition/multiplication)",
    "Integer Division Precision Loss (1/2 == 0)",
    "Floating Point Exact Equality (== instead of epsilon)",
    "Switch Case Fallthrough (Missing break)",

    # --- STRINGS & LIBC ---
    "String Literal Modification (Segfault)",
    "Strcmp Usage Error (Checking == 1 instead of == 0)",
    "Wrong Format Specifier (printf/scanf)",
    "Buffer Copy without Size Check (strcpy instead of strncpy)",

    # --- RECURSION & ALGORITHMS ---
    "Missing Base Case (Stack Overflow)",
    "Incorrect Base Case Return Value",
    "State Update Error in Recursion (Not passing modified args)",

    # --- TYPES & SYNTAX ---
    "Uninitialized Variable Usage",
    "Semicolon after Loop/If (Empty body)",
    "Operator Precedence Error",
    "Confusing Logical Operators (&& vs ||)"
]

# ==============================================================================
# 3. CORPUS D'EXERCICES
# ==============================================================================
EXERCISES_FULL = [
    # --- BASE ALGORITHMIC ---
    "Calculate Factorial (Iterative)", "Reverse a String (In-place)", "Find Maximum in Integer Array",
    "Convert Celsius to Fahrenheit", "Check for Palindrome String", "Swap Two Integers using Pointers",
    "Count Vowels in a String", "Dynamic Array Allocation and Fill", "Student Struct Average Calculation",
    "Manual String Copy (no strcpy)", "Multiplication Table (Nested Loops)", "Check for Prime Number",
    "Concatenate Two Strings (Dynamic Memory)", "Find Character Index in String", "Sum of Even Numbers in Array",
    "Convert Minutes to Hours:Minutes", "Pointer to Pointer (Double Indirection)", "Read Integer with Validation (scanf)",
    "Calculate Power (x^n)", "Bubble Sort Implementation", "Binary Search (Iterative)", "Fibonacci Series",
    "Leap Year Check", "Matrix Addition", "Transpose Matrix", "Count Words in String",
    "Find Duplicate in Array", "Rotate Array Left", "String Length Manual", "Check Armstrong Number",

    # --- STRING MANIPULATION (LIBC RE-IMPL) ---
    "Implement my_strlen (Recursive)", "Implement my_strcpy & my_strncpy", "Implement my_strcmp & my_strncmp",
    "Implement my_strchr (Find Char)", "Implement my_strstr (Needle in Haystack)", "Implement my_atoi (String to Int)",
    "Implement my_itoa (Int to String)", "String Capitalize (First Letter)",
    "Rot13 Cipher Implementation", "Remove Duplicate Chars from String", "Check Anagrams",
    "CamelCase to snake_case Converter",

    # --- MATH & ALGO ---
    "Greatest Common Divisor (Recursive)", "Least Common Multiple (LCM)", "Prime Factorization",
    "Sieve of Eratosthenes (Primes up to N)", "Square Root (Newton Method)", "Base Conversion (Decimal to Binary/Hex)",
    "Pascal's Triangle Generator", "Matrix Multiplication",
    "Find Missing Number in Sequence", "Merge Two Sorted Arrays", "Valid Parentheses Checker",

    # --- MEMORY & VOID* ---
    "My Malloc Wrapper (Simple Allocation)", "Implement my_memcpy", "Implement my_memmove (Handle Overlap)",
    "Implement my_memset", "Generic Swap Function (void*)", "Generic Array Linear Search (void*)",
    "2D Array Dynamic Allocation & Free", "Jagged Array Creation (Triangle)",

    # --- LINKED LISTS ---
    "Singly Linked List: Create Node", "Singly Linked List: Push Front/Back", "Singly Linked List: Pop Front/Back",
    "Singly Linked List: Get Size", "Singly Linked List: Get Element at Index", "Singly Linked List: Delete by Value",
    "Singly Linked List: Reverse (Iterative)", "Singly Linked List: Reverse (Recursive)", "Singly Linked List: Merge Sorted",
    "Detect Cycle in Linked List (Tortoise/Hare)", "Find Middle of Linked List",
    "Sort Linked List (Merge Sort)",

    # --- BITWISE ---
    "Print Binary Representation of Int", "Reverse Bits of a Byte", "Count Set Bits (Population Count)",
    "Swap Variables using XOR", "Check System Endianness", "Set, Unset, and Toggle Bits",
    "Extract RGB Values from Hex", "Is Power of Two (Bitwise)",

    # --- FILES & SYSTEM ---
    "Implement 'cat' Command", "Count Lines/Words/Chars in File",
    "Reverse File Content", "Implement 'tail' (Last N Lines)",

    # --- ADVANCED ---
    "Stack Implementation (Array-based)", "Queue Implementation (List-based)", "Evaluate RPN Expression (Stack)",
    "Towers of Hanoi Solver", "Flood Fill Algorithm", "Binary Search Tree: Insert Node",
    "Binary Search Tree: Search Value", "Binary Search Tree: In-order Traversal", "N-Queens Problem (Backtracking)",
]

# ==============================================================================
# 4. SYSTEM PROMPT RENFORCÉ
# ==============================================================================
SYSTEM_PROMPT = """You are an expert Computer Science Professor specializing in C programming and Semantic Code Analysis.
Your goal is to generate a high-quality "Contrastive Learning Dataset" for a major AI conference.

## THE TASK
Generate C code with ONE specific SEMANTIC error (the "Target Bug") while the rest of the logic remains correct.

## 🔴 ABSOLUTE REQUIREMENTS (MANDATORY)
1. **MUST COMPILE**: The code MUST compile successfully with: gcc -Wall -Wextra -o test test.c
   - NO syntax errors (missing semicolons, undeclared variables, missing brackets)
   - NO missing includes (stdio.h, stdlib.h, string.h, etc.)
   - All functions and variables must be properly declared

2. **SEMANTIC BUG ONLY**: The bug must be in LOGIC/BEHAVIOR, not syntax:
   ✅ VALID: Wrong condition (i <= n instead of i < n)
   ✅ VALID: Missing free() call
   ✅ VALID: Uninitialized variable
   ✅ VALID: Off-by-one error in array access
   ❌ INVALID: Missing semicolon
   ❌ INVALID: Undeclared variable
   ❌ INVALID: Missing #include

3. **SINGLE FAULT**: Only the requested bug. Rest of code must be perfect.

4. **REALISTIC**: Student-written style (use i, j, ptr, tmp, result, etc.)

5. **CAUSAL TEST FAILURES**: Failed tests must logically result from the bug.

## OUTPUT FORMAT
Return STRICTLY VALID JSON with these exact keys:
{
  "theme": "Exercise name",
  "difficulty": "beginner" | "intermediate" | "advanced",
  "tags": ["memory", "pointer", "recursion", etc.],
  "error_category": "Error type from taxonomy",
  "instructions": "Clear function description",
  "code": "Complete C code with includes",
  "test_cases_scope": ["Test 1 description", "Test 2", ...],
  "failed_tests": ["Test X: input=..., expected=..., actual=..."],
  "feedback": "Pedagogical hint (NOT the solution)"
}
"""

USER_PROMPT_TEMPLATE = """**Programming Exercise:** {topic}
**Deliberate SEMANTIC Bug:** {error_type}

Generate a complete C code sample that:
1. ✅ COMPILES successfully (gcc -Wall -Wextra)
2. ✅ Contains ONLY the specified semantic error: {error_type}
3. ✅ Uses realistic variable names (i, j, ptr, result, etc.)
4. ✅ Includes all necessary headers (stdio.h, stdlib.h, string.h, etc.)
5. ✅ Has test failures causally linked to the bug

**Classification:**
- Difficulty: beginner/intermediate/advanced (based on concept complexity)
- Tags: Add 2-4 relevant tags (memory, pointer, recursion, string, math, etc.)

**Return:** Valid JSON with keys: theme, difficulty, tags, error_category, instructions, code, test_cases_scope, failed_tests, feedback
"""

# ==============================================================================
# 5. VALIDATION FUNCTIONS
# ==============================================================================
def validate_compilation(code):
    """
    Valide que le code compile sans erreur.
    Returns: (success: bool, message: str)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        c_file = os.path.join(tmpdir, "test.c")
        exe_file = os.path.join(tmpdir, "test")

        try:
            with open(c_file, 'w', encoding='utf-8') as f:
                f.write(code)

            result = subprocess.run(
                ["gcc", "-Wall", "-Wextra", "-o", exe_file, c_file, "-lm"],
                capture_output=True,
                timeout=10,
                text=True
            )

            if result.returncode != 0:
                stderr = result.stderr
                if "error:" in stderr:
                    return False, f"COMPILATION ERROR: {stderr[:200]}"
                elif "warning:" in stderr:
                    return True, f"Compiled with warnings: {stderr[:100]}"
                else:
                    return False, f"Unknown compilation issue: {stderr[:200]}"

            return True, "Compiled successfully"

        except subprocess.TimeoutExpired:
            return False, "Compilation timeout"
        except Exception as e:
            return False, f"Validation error: {str(e)}"

def validate_semantic_bug(data):
    """
    Vérifie que le JSON contient bien un bug sémantique détectable.
    Returns: (valid: bool, message: str)
    """
    required_keys = ["theme", "difficulty", "tags", "error_category",
                     "instructions", "code", "test_cases_scope", "failed_tests", "feedback"]

    if not all(k in data for k in required_keys):
        missing = [k for k in required_keys if k not in data]
        return False, f"Missing keys: {missing}"

    failed_str = str(data.get("failed_tests", "")).lower()
    suspicious_success = [
        "(correct)", "all pass", "outputs '", "outputs \"",
        "test passed", "correct output"
    ]

    suspect_count = sum(1 for pattern in suspicious_success if pattern in failed_str)
    if suspect_count >= 2:
        return False, f"Suspicious: tests seem to PASS ({suspect_count} success patterns found)"

    failed_tests = data.get("failed_tests", [])
    if not failed_tests:
        return False, "No failed tests specified"

    if isinstance(failed_tests, str):
        failed_tests = [failed_tests]

    if len(failed_tests) == 0:
        return False, "Empty failed_tests list"

    if data.get("difficulty") not in ["beginner", "intermediate", "advanced"]:
        return False, f"Invalid difficulty: {data.get('difficulty')}"

    tags = data.get("tags", [])
    if not isinstance(tags, list) or len(tags) < 1:
        return False, "Tags must be a non-empty list"

    return True, "Semantic validation passed"

# ==============================================================================
# 6. LOGGING DES MÉTRIQUES
# ==============================================================================
def log_metrics(metrics_data):
    """
    Enregistre les métriques de génération dans un fichier JSONL.
    """
    with METRICS_LOCK:
        with open(METRICS_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics_data, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())

def update_global_stats(metrics):
    """
    Met à jour les statistiques globales (thread-safe).
    """
    with METRICS_LOCK:
        GLOBAL_STATS["total_attempts"] += metrics["attempts"]

        if metrics["success"]:
            GLOBAL_STATS["total_successes"] += 1
        else:
            GLOBAL_STATS["total_failures"] += 1
            reason = metrics.get("failure_reason", "unknown")
            GLOBAL_STATS["failure_reasons"][reason] = GLOBAL_STATS["failure_reasons"].get(reason, 0) + 1

        GLOBAL_STATS["total_tokens_prompt"] += metrics["total_tokens_prompt"]
        GLOBAL_STATS["total_tokens_completion"] += metrics["total_tokens_completion"]
        GLOBAL_STATS["total_tokens_total"] += metrics["total_tokens_total"]
        GLOBAL_STATS["total_time_seconds"] += metrics["generation_time_seconds"]

# ==============================================================================
# 7. GÉNÉRATION AVEC MÉTRIQUES
# ==============================================================================
def generate_synthetic_entry(exercise):
    """
    Génère une entrée avec validation stricte + métriques détaillées.
    """
    selected_error = random.choice(ERROR_TAXONOMY)
    prompt = USER_PROMPT_TEMPLATE.format(topic=exercise, error_type=selected_error)

    # Métriques pour ce sample
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "exercise": exercise,
        "error_type": selected_error,
        "attempts": 0,
        "success": False,
        "failure_reason": None,
        "total_tokens_prompt": 0,
        "total_tokens_completion": 0,
        "total_tokens_total": 0,
        "generation_time_seconds": 0,
        "retry_details": []
    }

    start_time = time.time()

    for attempt in range(MAX_RETRIES):
        metrics["attempts"] += 1
        retry_info = {
            "attempt_number": attempt + 1,
            "tokens_prompt": 0,
            "tokens_completion": 0,
            "validation_result": None,
            "compilation_result": None
        }

        try:
            # Appel API
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.85,
                max_tokens=1500
            )

            content = response.choices[0].message.content

            # Capturer les tokens (usage de l'API)
            usage = response.usage
            retry_info["tokens_prompt"] = usage.prompt_tokens
            retry_info["tokens_completion"] = usage.completion_tokens

            metrics["total_tokens_prompt"] += usage.prompt_tokens
            metrics["total_tokens_completion"] += usage.completion_tokens
            metrics["total_tokens_total"] += usage.total_tokens

            # Parse JSON
            data = json.loads(content)

            # VALIDATION NIVEAU 1: Structure + Sémantique
            valid, msg = validate_semantic_bug(data)
            retry_info["validation_result"] = {"valid": valid, "message": msg}

            if not valid:
                print(f"  ❌ [{exercise}] Semantic validation failed (attempt {attempt+1}/{MAX_RETRIES}): {msg}")
                metrics["retry_details"].append(retry_info)
                continue

            # VALIDATION NIVEAU 2: Compilation
            code = data.get("code", "")
            compiles, compile_msg = validate_compilation(code)
            retry_info["compilation_result"] = {"compiled": compiles, "message": compile_msg}

            if not compiles:
                print(f"  ❌ [{exercise}] Compilation failed (attempt {attempt+1}/{MAX_RETRIES}): {compile_msg}")
                metrics["retry_details"].append(retry_info)
                continue

            # ✅ SUCCÈS
            metrics["success"] = True
            metrics["generation_time_seconds"] = time.time() - start_time
            metrics["retry_details"].append(retry_info)

            print(f"  ✅ [{exercise}] Valid sample generated ({compile_msg})")

            # Logger les métriques
            log_metrics(metrics)
            update_global_stats(metrics)

            return data

        except json.JSONDecodeError:
            retry_info["validation_result"] = {"valid": False, "message": "Invalid JSON"}
            metrics["retry_details"].append(retry_info)
            print(f"  ⚠️  [{exercise}] Invalid JSON (attempt {attempt+1}/{MAX_RETRIES})")

        except Exception as e:
            err_msg = str(e)
            retry_info["validation_result"] = {"valid": False, "message": f"API Error: {err_msg}"}
            metrics["retry_details"].append(retry_info)

            if "429" in err_msg or "Rate limit" in err_msg:
                sleep_time = 10 * (attempt + 1)
                print(f"  ⏸️  Rate limit hit. Sleeping {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                print(f"  💥 [{exercise}] API Error: {err_msg}")
                break

    # Échec après MAX_RETRIES tentatives
    metrics["success"] = False
    metrics["generation_time_seconds"] = time.time() - start_time
    metrics["failure_reason"] = "max_retries_exceeded"

    # Déterminer la raison principale d'échec
    if metrics["retry_details"]:
        last_retry = metrics["retry_details"][-1]
        if last_retry.get("compilation_result") and not last_retry["compilation_result"]["compiled"]:
            metrics["failure_reason"] = "compilation_error"
        elif last_retry.get("validation_result") and not last_retry["validation_result"]["valid"]:
            metrics["failure_reason"] = "semantic_validation_failed"

    print(f"  ⛔ [{exercise}] SKIPPED after {MAX_RETRIES} failed attempts")

    log_metrics(metrics)
    update_global_stats(metrics)

    return None

# ==============================================================================
# 8. MAIN EXECUTION
# ==============================================================================
def save_summary():
    """
    Sauvegarde un résumé complet de la génération.
    """
    GLOBAL_STATS["end_time"] = datetime.now().isoformat()

    if GLOBAL_STATS["start_time"]:
        start = datetime.fromisoformat(GLOBAL_STATS["start_time"])
        end = datetime.fromisoformat(GLOBAL_STATS["end_time"])
        total_duration = (end - start).total_seconds()
        GLOBAL_STATS["total_duration_seconds"] = total_duration
        GLOBAL_STATS["total_duration_human"] = f"{int(total_duration // 3600)}h {int((total_duration % 3600) // 60)}m"

    # Calculer les coûts (estimation DeepSeek: ~$0.14/1M input, ~$0.28/1M output)
    cost_input = (GLOBAL_STATS["total_tokens_prompt"] / 1_000_000) * 0.14
    cost_output = (GLOBAL_STATS["total_tokens_completion"] / 1_000_000) * 0.28
    GLOBAL_STATS["estimated_cost_usd"] = round(cost_input + cost_output, 2)

    # Success rate
    total_tasks = GLOBAL_STATS["total_successes"] + GLOBAL_STATS["total_failures"]
    if total_tasks > 0:
        GLOBAL_STATS["success_rate"] = round(100 * GLOBAL_STATS["total_successes"] / total_tasks, 2)

    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(GLOBAL_STATS, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 70)
    print("📊 RÉSUMÉ DE LA GÉNÉRATION")
    print("=" * 70)
    print(f"✅ Succès : {GLOBAL_STATS['total_successes']}")
    print(f"⛔ Échecs : {GLOBAL_STATS['total_failures']}")
    print(f"📈 Success Rate : {GLOBAL_STATS.get('success_rate', 0)}%")
    print(f"🔢 Total tentatives : {GLOBAL_STATS['total_attempts']}")
    print(f"🎫 Tokens prompt : {GLOBAL_STATS['total_tokens_prompt']:,}")
    print(f"🎫 Tokens completion : {GLOBAL_STATS['total_tokens_completion']:,}")
    print(f"🎫 Tokens total : {GLOBAL_STATS['total_tokens_total']:,}")
    print(f"💰 Coût estimé : ${GLOBAL_STATS.get('estimated_cost_usd', 0)}")
    print(f"⏱️  Durée totale : {GLOBAL_STATS.get('total_duration_human', 'N/A')}")
    print()
    print("🔍 Raisons d'échec :")
    for reason, count in GLOBAL_STATS["failure_reasons"].items():
        print(f"   {count:4}× {reason}")
    print("=" * 70)

def main():
    GLOBAL_STATS["start_time"] = datetime.now().isoformat()

    print("=" * 70)
    print("🚀 GÉNÉRATEUR DE DATASET VALIDÉ (Avec métriques détaillées)")
    print("=" * 70)

    # État initial
    existing_count = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            existing_count = sum(1 for line in f)

    to_generate = TOTAL_SAMPLES_TARGET - existing_count
    print(f"📊 Déjà générés : {existing_count}")
    print(f"🎯 Cible restante : {to_generate}")
    print(f"⚙️  Max retries par sample : {MAX_RETRIES}")
    print(f"👷 Workers concurrents : {MAX_WORKERS}")
    print(f"📁 Métriques : {METRICS_FILE}")
    print()

    if to_generate <= 0:
        print("✅ Objectif atteint. Arrêt.")
        return

    # Préparation des tâches
    tasks = []
    while len(tasks) < to_generate:
        batch = EXERCISES_FULL.copy()
        random.shuffle(batch)
        tasks.extend(batch)
    tasks = tasks[:to_generate]

    # Exécution
    print(f"🏃 Démarrage de la génération avec validation...")
    print()

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_task = {executor.submit(generate_synthetic_entry, task): task for task in tasks}

            pbar = tqdm(as_completed(future_to_task), total=len(tasks), unit="sample", ncols=80)

            success_count = 0
            skipped_count = 0

            for future in pbar:
                result = future.result()
                if result:
                    with FILE_LOCK:
                        f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f_out.flush()
                        os.fsync(f_out.fileno())
                    success_count += 1
                else:
                    skipped_count += 1

                pbar.set_description(f"✅ {success_count} | ⛔ {skipped_count}")

    save_summary()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Arrêt d'urgence. Sauvegarde des métriques...")
        save_summary()
