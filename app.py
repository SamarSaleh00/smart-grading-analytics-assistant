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

GRADING_KEYWORDS = [
    "quiz", "assignment", "exam", "score", "grade", "grades", "student", "students",
    "average", "pattern", "common", "missing", "submitted", "late", "cohort"
]

GREETINGS = [
    "hi", "hello", "hey", "hola", "salam", "yo", "morning", "good morning",
    "good afternoon", "good evening", "how are you", "what's up", "thanks",
    "thank you", "bye", "goodbye"
]

META_PATTERNS = [
    "what do you mean", "what does that mean", "i don't understand", "i dont understand",
    "can you clarify", "can you explain", "explain that", "what do u mean", "not clear"
]

def is_smalltalk(query):
    """Catches greetings, thanks, vague chatter, and meta-questions about a prior answer,
    none of which should be routed to the table QA engine. TAPAS always returns *some*
    cell as an answer even for nonsense input, so anything that isn't clearly a new
    grading question needs to be filtered out before it gets there."""
    lowered = query.strip().lower()
    if not lowered:
        return True
    if any(p in lowered for p in META_PATTERNS):
        return True
    has_grading_keyword = any(k in lowered for k in GRADING_KEYWORDS)
    is_greeting = any(lowered == g or lowered.startswith(g) for g in GREETINGS)
    has_digit = any(ch.isdigit() for ch in lowered)
    # Short, keyword-free, digit-free input is treated as chit-chat rather than a data query
    is_vague = len(lowered.split()) <= 5 and not has_grading_keyword and not has_digit
    return is_greeting or is_vague

def smalltalk_response(query):
    lowered = query.strip().lower()
    if any(p in lowered for p in META_PATTERNS):
        last_answer = next(
            (e["content"] for e in reversed(st.session_state.chat_history) if e["role"] == "assistant"),
            None
        )
        if last_answer:
            return f"To clarify my last answer: {last_answer} Let me know if you'd like it broken down differently."
        return "I don't have a previous answer to clarify yet — what would you like to know about the class data?"
    if any(g in lowered for g in ["thanks", "thank you"]):
        return "You're welcome! Let me know if you'd like to check another student or metric."
    if any(g in lowered for g in ["bye", "goodbye"]):
        return "Goodbye! Come back anytime you need a grade breakdown."
    return (
        "Hi! I'm your grading assistant. Ask me things like "
        "\"Which students scored below 60% on the final exam?\" or "
        "\"What's the average Quiz 1 score?\" and I'll pull it from the class data. "
        "(Right now I only understand questions written in English.)"
    )

def check_clarification_gate(query):
    lowered = query.lower()
    # Map ordinal words to digits so "first quiz" is recognized the same as "Quiz 1"
    ordinal_map = {"first": "1", "second": "2", "third": "3"}
    for word, digit in ordinal_map.items():
        if word in lowered:
            lowered += f" {digit}"
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
        coordinates = result.get("coordinates") or []

        if not coordinates:
            return (
                "I couldn't match that to a specific column or student in the data. "
                "Try asking about a specific quiz, assignment, or exam score."
            )

        if not is_follow_up:
            selected_rows = list(set([coord[0] for coord in coordinates]))
            st.session_state.active_subset = active_df.iloc[selected_rows].copy()

        cells = result.get("cells", [])
        aggregator = (result.get("aggregator") or "NONE").upper()

        # TAPAS only labels which operator applies (SUM/AVERAGE/COUNT) — it doesn't
        # compute the value itself. If it labeled numeric cells with an aggregator,
        # do the actual math here instead of printing the raw "AVERAGE > ..." string.
        if aggregator in ("SUM", "AVERAGE", "COUNT") and cells:
            try:
                numeric_cells = [float(c) for c in cells]
            except ValueError:
                numeric_cells = None

            if numeric_cells:
                if aggregator == "AVERAGE":
                    value = sum(numeric_cells) / len(numeric_cells)
                    return f"The average is {value:.1f} (based on {len(numeric_cells)} matching values)."
                if aggregator == "SUM":
                    return f"The total is {sum(numeric_cells):.1f}."
                if aggregator == "COUNT":
                    return f"There are {len(numeric_cells)} matching entries."
            # Aggregator was mislabeled on non-numeric cells (e.g. names) — just list them
            return ", ".join(cells)

        ans = result.get("answer", "")
        if not ans and cells:
            ans = ", ".join(cells)
        return ans if ans else "I couldn't find matching items."
    except Exception:
        return "Something went wrong querying the student data. Try rephrasing your question."

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
            # 0. Filter out greetings/chit-chat before touching the data engine
            if is_smalltalk(user_input):
                response = smalltalk_response(user_input)
            else:
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
