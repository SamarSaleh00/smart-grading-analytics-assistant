import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Set page configuration to wide layout for side-by-side view
st.set_page_config(layout="wide", page_title="AI Educational Assistant", page_icon="🤖")

# --- PHASE 1: DATA ENGINE & MODEL CACHING ---

@st.cache_data
def ensure_synthetic_data():
    """Generates the mock grading dataset if it doesn't exist."""
    np.random.seed(42)
    names = [
        "Alex", "Omar", "Sarah", "John", "Emily", "Michael", "Jessica", "David",
        "Taylor", "James", "Emma", "Daniel", "Olivia", "William", "Sophia",
        "Liam", "Ava", "Noah", "Isabella", "Lucas"
    ]
    data = {
        "Student_ID": list(range(1001, 1021)),
        "Student_Name": names,
        "Quiz_1_Score": np.random.randint(70, 98, size=20),
        "Quiz_2_Score": np.random.randint(68, 100, size=20),
        "Assignment_1_Score": np.random.randint(75, 100, size=20),
        "Assignment_2_Score": np.random.randint(72, 100, size=20),
        "Assignment_3_Status": np.random.choice(["Submitted", "Late", "Missing"], size=20, p=[0.70, 0.15, 0.15]),
        "Final_Exam_Score": np.random.randint(65, 100, size=20)
    }
    df = pd.DataFrame(data)

    # Inject specific testing anomalies for Alex, Omar, and Sarah
    target_struggling = ["Alex", "Omar", "Sarah"]
    for name in target_struggling:
        df.loc[df["Student_Name"] == name, "Final_Exam_Score"] = np.random.randint(45, 58)
        df.loc[df["Student_Name"] == name, "Assignment_3_Status"] = "Missing"
        df.loc[df["Student_Name"] == name, "Assignment_1_Score"] = np.random.randint(55, 68)
        df.loc[df["Student_Name"] == name, "Assignment_2_Score"] = np.random.randint(50, 65)
    return df

@st.cache_resource
def load_ai_models():
    """Loads and caches models on the server to prevent reload lag."""
    # Load Query Rewriter
    model_id = "google/flan-t5-small"
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    # Load Table QA Pipeline
    tapa = pipeline("table-question-answering", model="google/tapas-base-finetuned-wtq")
    return tok, mod, tapa

# Initialize session structures
if "df" not in st.session_state:
    st.session_state.df = ensure_synthetic_data()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_subset" not in st.session_state:
    st.session_state.active_subset = None

tokenizer, rewriter_model, table_qa_pipeline = load_ai_models()

# --- PHASE 2 & 4: NLU, DST, & DIALOGUE POLICY MODULES ---

def get_history_string():
    history_str = ""
    # Look at the last 3 turns in state
    for entry in st.session_state.chat_history[-6:]:
        prefix = "User" if entry["role"] == "user" else "Assistant"
        history_str += f"{prefix}: {entry['content']}\n"
    return history_str.strip()

def rewrite_query(current_query):
    history = get_history_string()
    if not history:
        return current_query
    prompt = (
        "Given the following conversation history, rewrite the user's latest query to be "
        "fully independent and explicit. Replace pronouns like 'they', 'their', or 'it' "
        f"with the exact subject from history.\n\nHistory:\n{history}\n\nQuery: {current_query}\n\nRewritten:"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = rewriter_model.generate(**inputs, max_new_tokens=50, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def check_clarification_gate(query):
    lowered = query.lower()
    if "quiz" in lowered and not any(x in lowered for x in ["1", "2", "one", "two"]):
        return "Could you please specify if you mean Quiz 1 or Quiz 2?"
    if "assignment" in lowered and not any(x in lowered for x in ["1", "2", "3", "one", "two", "three", "average", "all"]):
        return "Could you please specify if you mean Assignment 1, Assignment 2, or Assignment 3?"
    return None

def analyze_anomalies(subset):
    if subset is None or subset.empty:
        return None
    missing_a3 = len(subset[subset["Assignment_3_Status"] == "Missing"])
    if (missing_a3 / len(subset)) > 0.50:
        pct = int((missing_a3 / len(subset)) * 100)
        return f"Yes, {pct}% of these students have not submitted Assignment 3. Would you like me to draft a reminder email for them?"
    return None

# --- PHASE 3: STATEFUL RETRIEVAL ENGINE ---

def execute_table_qa(query):
    is_follow_up = any(w in query.lower() for w in ["their", "them", "they", "average", "pattern", "this"])

    if is_follow_up and st.session_state.active_subset is not None and not st.session_state.active_subset.empty:
        active_df = st.session_state.active_subset
    else:
        active_df = st.session_state.df

    tapas_table = active_df.astype(str)
    try:
        result = table_qa_pipeline(table=tapas_table, query=query)
        if result.get("coordinates") and not is_follow_up:
            selected_rows = list(set([coord[0] for coord in result["coordinates"]]))
            st.session_state.active_subset = active_df.iloc[selected_rows].copy()

        ans = result.get("answer", "")
        if not ans and result.get("cells"):
            ans = ", ".join(result["cells"])
        return ans if ans else "I couldn't find matching items."
    except Exception:
        return "Error querying the table layout."

# --- APPLICATION UI LAYOUT ---

# Layout split: Left panel for live database monitoring, Right panel for Chat
col_dash, col_chat = st.columns([1, 1.2])

with col_dash:
    st.header("📊 Live Student Analytics")
    if st.session_state.active_subset is not None:
        st.subheader("🎯 Isolated Focus Cohort")
        st.dataframe(st.session_state.active_subset, use_container_width=True)
        if st.button("Reset Data Focus to Entire Class"):
            st.session_state.active_subset = None
            st.rerun()
    else:
        st.subheader("🏫 Full Class Database")
        st.dataframe(st.session_state.df, use_container_width=True)

with col_chat:
    st.header("🤖 Conversational Assistant")

    # Display the active conversation log
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_input := st.chat_input("Ask about grades, averages, or student progress trends..."):
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("Processing performance layers..."):
            # 1. Check Guardrails
            response = check_clarification_gate(user_input)

            if not response:
                lowered_in = user_input.lower()
                # 2. Logic Orchestration & Analytics Fallbacks
                if "pattern" in lowered_in or "common" in lowered_in:
                    insight = analyze_anomalies(st.session_state.active_subset)
                    response = insight if insight else "No major overlapping behavioral anomalies caught in this group."
                elif "average" in lowered_in and "assignment" in lowered_in:
                    if st.session_state.active_subset is not None:
                        a1_m = st.session_state.active_subset["Assignment_1_Score"].mean()
                        a2_m = st.session_state.active_subset["Assignment_2_Score"].mean()
                        response = f"The average assignment score for this group is {((a1_m + a2_m)/2):.1f}% (A1: {a1_m:.1f}%, A2: {a2_m:.1f}%)."
                    else:
                        response = "Please isolate a student cohort first so I know whose averages to calculate."
                else:
                    # 3. Handle via Query Rewriter and TAPAS
                    rewritten = rewrite_query(user_input)
                    response = execute_table_qa(rewritten)

        with st.chat_message("assistant"):
            st.write(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
