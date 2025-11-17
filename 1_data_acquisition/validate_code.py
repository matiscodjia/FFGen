import json
import subprocess
import tempfile
import os
import argparse
import sys
from tqdm import tqdm # Pour une belle barre de progression

def compile_c_snippet(snippet_text: str) -> str:
    """
    Tente de compiler un extrait de code C et retourne la sortie du compilateur (erreurs/warnings).
    
    Args:
        snippet_text: La chaîne de caractères contenant le code C.

    Returns:
        La sortie de stderr/stdout du compilateur, ou un message d'erreur.
    """
    
    # Créer un fichier temporaire avec l'extension .c pour que gcc le reconnaisse
    try:
        with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode='w', encoding='utf-8') as tmp_file:
            tmp_file.write(snippet_text)
            tmp_file_path = tmp_file.name
        
        # Commande de compilation
        # -c : Compile mais ne fait pas l'édition de liens (pas besoin de main())
        # -Wall -Wextra : Active tous les avertissements (très utile)
        # -o /dev/null : Jette le fichier objet de sortie (NUL sur Windows)
        output_dest = "NUL" if os.name == 'nt' else "/dev/null"
        command = [
            'gcc', 
            '-Wall', 
            '-Wextra', 
            '-c', 
            '-o', output_dest,
            tmp_file_path
        ]

        # Exécuter la commande
        # Nous capturons stderr car gcc écrit les erreurs et warnings ici.
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        
        # Combiner stderr et stdout (au cas où)
        compiler_msg = result.stderr + result.stdout
        
        if not compiler_msg:
            return "OK" # Pas de messages, compilation réussie

        return compiler_msg.strip()

    except FileNotFoundError:
        return "ERREUR_INTERNE: 'gcc' n'est pas installé ou n'est pas dans le PATH."
    except Exception as e:
        return f"ERREUR_INTERNE: Échec de l'exécution du subprocess: {e}"
    finally:
        # S'assurer que le fichier temporaire est supprimé
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

def process_jsonl_file(input_path: str, output_path: str):
    """
    Lit un fichier jsonl, compile chaque 'code_snippet', et écrit un nouveau jsonl.
    """
    try:
        # Compter le nombre de lignes pour tqdm
        with open(input_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for line in f)

        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            print(f"Compilation des snippets de {input_path}...")
            
            # Utiliser tqdm pour la progression
            for line in tqdm(infile, total=line_count, desc="Compilation..."):
                try:
                    data = json.loads(line)
                    
                    snippet = data.get('code_snippet')
                    
                    if snippet:
                        compiler_msg = compile_c_snippet(snippet)
                        data['compiler_msg'] = compiler_msg
                    else:
                        data['compiler_msg'] = "ERREUR: Pas de 'code_snippet' dans cet objet."
                        
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    
                except json.JSONDecodeError:
                    print(f"Avertissement: Ligne invalide (non-JSON) ignorée: {line[:50]}...")
                    
    except FileNotFoundError:
        print(f"Erreur: Le fichier d'entrée '{input_path}' n'a pas été trouvé.", file=sys.stderr)
    except Exception as e:
        print(f"Une erreur inattendue est survenue: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Compile des extraits de code C depuis un fichier JSONL.")
    parser.add_argument("input_file", help="Le chemin vers le fichier d'entrée .jsonl")
    parser.add_argument("output_file", help="Le chemin vers le fichier de sortie .jsonl où sauvegarder les résultats")
    
    args = parser.parse_args()
    
    process_jsonl_file(args.input_file, args.output_file)
    print(f"\nTerminé ! Résultats sauvegardés dans {args.output_file}")

if __name__ == "__main__":
    main()
