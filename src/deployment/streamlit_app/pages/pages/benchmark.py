import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
st.set_page_config(layout="wide", page_title="FFGen - Dataset Benchmark")

# File paths
DATASET_SOURCE = "datasets/dataset_c_piscine_semantic.jsonl" 
OUTPUT_CSV = "benchmarks/human_evaluation_results.csv"

# Ensure directories exist
os.makedirs("benchmarks", exist_ok=True)
os.makedirs("datasets", exist_ok=True)

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

@st.cache_data
def load_dataset_sample(filepath, sample_size=100):
    """Loads a sample of the dataset for evaluation"""
    data = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i >= sample_size: break
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except: continue
    else:
        st.error(f"File not found: {filepath}")
        return pd.DataFrame()
    return pd.DataFrame(data)

def save_evaluation(eval_data):
    """Saves a single evaluation line"""
    df = pd.DataFrame([eval_data])
    if not os.path.exists(OUTPUT_CSV):
        df.to_csv(OUTPUT_CSV, index=False)
    else:
        df.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)

# ==========================================
# STATE MANAGEMENT (SESSION STATE)
# ==========================================
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0

# Load data
df = load_dataset_sample(DATASET_SOURCE)

# ==========================================
# MAIN INTERFACE
# ==========================================

st.title("FFGen Benchmark: Dataset Quality Audit")
st.markdown("""
**Evaluation Protocol:**
This interface aims to establish an objective quality score for the dataset. 
Please evaluate each Code/Feedback pair according to the strict criteria below.
""")

if not df.empty:
    # Progress bar
    progress = st.session_state.current_idx / len(df)
    st.progress(progress, text=f"Progress: {st.session_state.current_idx + 1}/{len(df)}")

    # Get current entry
    if st.session_state.current_idx < len(df):
        row = df.iloc[st.session_state.current_idx]
        
        # --- LAYOUT: 2 COLUMNS ---
        col_data, col_eval = st.columns([1.2, 0.8])

        # --- LEFT COLUMN: DATA TO EVALUATE ---
        with col_data:
            st.subheader("🔍 Dataset Entry")
            
            # Context
            with st.expander("View exercise context", expanded=False):
                st.markdown(f"**Theme:** {row.get('theme', 'N/A')}")
                st.markdown(f"**Simulated Error:** {row.get('error_category', 'N/A')}")
                st.text(row.get('instructions', ''))

            # Code
            st.markdown("### Student Code (Input)")
            st.code(row.get('code', ''), language='c')

            # Feedback
            st.markdown("### AI Feedback (Output)")
            st.info(row.get('feedback', ''))

        # --- RIGHT COLUMN: EVALUATION GRID ---
        with col_eval:
            st.subheader("Audit Grid")
            
            with st.form("benchmark_form"):
                
                # CRITERION 1: TECHNICAL VALIDITY (Factual)
                st.markdown("#### 1. Technical Validity")
                technical_validity = st.radio(
                    "Does the feedback describe a correct technical reality in C?",
                    options=["Yes (Correct)", "No (Hallucination/False)", "Debatable (Imprecise)"],
                    index=None,
                    help="If the AI mentions a non-existent function or incorrect memory behavior, select NO."
                )

                # CRITERION 2: DIAGNOSTIC (Factual)
                st.markdown("#### 2. Diagnostic Precision")
                diagnostic_quality = st.radio(
                    "Does the feedback identify the root cause?",
                    options=[
                        "Root Cause (e.g., missing malloc)", 
                        "Symptom Only (e.g., infinite loop/memory leak)", 
                        "Off-topic"
                    ],
                    index=None
                )

                # CRITERION 3: PEDAGOGY (Epitech Method)
                st.markdown("#### 3. Pedagogical Compliance")
                
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    has_spoiler = st.checkbox("Contains solution (Fixed code/Spoiler)")
                with col_c2:
                    is_maieutic = st.checkbox("Asks a question or uses a guiding concept")

                # CRITERION 4: ACTIONABILITY
                st.markdown("#### 4. Actionability")
                actionability = st.select_slider(
                    "Can a beginner student fix their code using this feedback?",
                    options=["Impossible", "Difficult", "Doable", "Immediate"]
                )

                expert_name = st.text_input("Evaluator Name", value="Expert_1")
                comment = st.text_area("Specific comment (optional)")

                # SUBMISSION BUTTONS
                st.divider()
                submitted = st.form_submit_button("💾 Save Evaluation", type="primary")
                
                if submitted:
                    if technical_validity is None or diagnostic_quality is None:
                        st.error("Technical and diagnostic criteria are mandatory.")
                    else:
                        # Create data object
                        eval_entry = {
                            "dataset_id": row.get('id', st.session_state.current_idx),
                            "evaluator": expert_name,
                            "timestamp": datetime.now().isoformat(),
                            "tech_validity": technical_validity,
                            "diagnostic_score": diagnostic_quality,
                            "has_spoiler": has_spoiler,
                            "is_maieutic": is_maieutic,
                            "actionability": actionability,
                            "comment": comment
                        }
                        
                        save_evaluation(eval_entry)
                        st.success("Validated!")
                        st.session_state.current_idx += 1
                        st.rerun()

    else:
        st.success("Audit complete for this sample!")
        
        # Immediate summary dashboard
        if os.path.exists(OUTPUT_CSV):
            res_df = pd.read_csv(OUTPUT_CSV)
            st.divider()
            st.subheader("Preliminary Results")
            
            c1, c2, c3 = st.columns(3)
            
            # Calculate validity rate
            # Note: We check if string contains "Yes" to match the English option
            valid_count = res_df[res_df['tech_validity'].str.contains("Yes")].shape[0]
            valid_rate = (valid_count / len(res_df)) * 100
            c1.metric("Technical Validity", f"{valid_rate:.1f}%")
            
            # Calculate spoiler rate
            spoil_count = res_df[res_df['has_spoiler'] == True].shape[0]
            spoil_rate = (spoil_count / len(res_df)) * 100
            c2.metric("Spoiler Rate (Should be low)", f"{spoil_rate:.1f}%", delta_color="inverse")
            
            c3.metric("Audited Samples", len(res_df))

else:
    st.warning(f"No dataset found at location: `{DATASET_SOURCE}`. Please check the path.")