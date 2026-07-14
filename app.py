import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import operator
from groq import Groq

st.set_page_config(layout="wide", page_title="AI Educational Assistant", page_icon="🤖")

GROQ_MODEL = "llama-3.1-8b-instant"

# =========================================================================
# PHASE 1: DATA ENGINE
# =========================================================================

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

    target_struggling = ["Alex", "Omar", "Sarah"]
    for name in target_struggling:
        df.loc[df["Student_Name"] == name, "Final_Exam_Score"] = np.random.randint(45, 58)
        df.loc[df["Student_Name"] == name, "Assignment_3_Status"] = "Missing"
        df.loc[df["Student_Name"] == name, "Assignment_1_Score"] = np.random.randint(55, 68)
        df.loc[df["Student_Name"] == name, "Assignment_2_Score"] = np.random.randint(50, 65)
    return df

DATA_COLUMNS_NUMERIC = [
    "Quiz_1_Score", "Quiz_2_Score", "Assignment_1_Score",
    "Assignment_2_Score", "Final_Exam_Score"
]
DATA_COLUMNS_ALL = DATA_COLUMNS_NUMERIC + ["Assignment_3_Status"]

# =========================================================================
# PHASE 2: LLM CLIENT (replaces TAPAS + FLAN-T5-small)
# =========================================================================

def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")

@st.cache_resource
def get_groq_client():
    api_key = get_api_key()
    if not api_key:
        st.error(
            "No GROQ_API_KEY found. Add it to `.streamlit/secrets.toml` as "
            "`GROQ_API_KEY = \"...\"` or set it as an environment variable, "
            "then restart the app."
        )
        st.stop()
    return Groq(api_key=api_key)

client = get_groq_client()

def call_llm(messages, temperature=0.2, max_tokens=400, json_mode=False):
    """Single choke point for every LLM call in the app."""
    kwargs = dict(model=GROQ_MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content
    except Exception as e:
        return json.dumps({"error": str(e)}) if json_mode else f"(LLM error: {e})"

def is_arabic(text):
    arabic_chars = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    return arabic_chars > max(1, len(text.strip())) * 0.3

def lang_name(code):
    return "Arabic" if code == "ar" else "English"

# =========================================================================
# PHASE 3: NLU — intent classification + coreference resolution + slot filling
# One structured LLM call replaces: FLAN-T5 query rewriting, the English-only
# keyword router, and the hardcoded English clarification gate.
# =========================================================================

NLU_SYSTEM_PROMPT = f"""You are the NLU (Natural Language Understanding) module of a conversational \
grading assistant used by a teacher. The teacher may write in English or Arabic, and may switch \
between them at any point.

The student data table has these columns: {', '.join(DATA_COLUMNS_ALL)} (plus Student_Name).
Assignment_3_Status is categorical with values: Submitted, Late, Missing.
All other score columns are numeric (0-100).

Given the recent conversation history and the latest user message, output ONLY a single JSON object \
(no markdown, no commentary) with these fields:

{{
  "language": "ar" or "en" (language of the LATEST user message),
  "intent": one of ["smalltalk", "clarify", "meta_clarify", "reset_focus", "data_filter",
                     "data_aggregate", "data_lookup", "pattern_analysis", "general_knowledge"],
  "standalone_query_en": "the user's request rewritten in English as a fully self-contained question, \
resolving any pronoun or reference (e.g. 'their', 'them', 'هم', 'هذا') back to whatever it refers to \
in the conversation history",
  "column": "the single best-matching column name from the list above, or null",
  "student_name": "a specific student's name if the user is asking about one student, or null",
  "filter_operator": one of "<", "<=", ">", ">=", "==", "!=" or null,
  "filter_value": "the comparison value as a string, or null",
  "aggregate_function": one of "mean", "sum", "count", "min", "max" or null,
  "clarification_question": "if intent is 'clarify', a short question IN THE SAME LANGUAGE as the \
user's message asking them to specify which quiz/assignment they mean; else null"
}}

Guidance:
- "data_filter": the user wants a list/subset of students matching a condition (e.g. below a score, missing an assignment).
- "data_aggregate": the user wants a single number (average, sum, count, min, max) over a column.
- "data_lookup": the user wants a specific value(s), often for one named student.
- "pattern_analysis": the user asks about patterns, common issues, or shared risk factors in the current group.
- "reset_focus": the user asks to go back to / show the whole class again.
- "clarify": the user mentions "quiz" without saying which one, or "assignment" without saying which \
one (or "average"/"all"), and you cannot safely guess.
- "meta_clarify": the user is confused about or asking you to re-explain your OWN previous answer.
- "smalltalk": greetings, thanks, goodbyes, or empty/very vague chatter with no data intent.
- "general_knowledge": anything else — general questions, advice, requests unrelated to the class data. \
This assistant should still answer these helpfully; do not force them into a data intent.

Output ONLY the JSON object."""

def run_nlu(user_input, history_str):
    messages = [
        {"role": "system", "content": NLU_SYSTEM_PROMPT},
        {"role": "user", "content": f"Conversation history:\n{history_str or '(none)'}\n\nLatest user message:\n{user_input}"}
    ]
    raw = call_llm(messages, temperature=0.0, max_tokens=400, json_mode=True)
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parsed = {}
    # Safe defaults so downstream code never KeyErrors
    defaults = {
        "language": "ar" if is_arabic(user_input) else "en",
        "intent": "general_knowledge",
        "standalone_query_en": user_input,
        "column": None, "student_name": None,
        "filter_operator": None, "filter_value": None,
        "aggregate_function": None, "clarification_question": None,
    }
    defaults.update({k: v for k, v in parsed.items() if v is not None})
    return defaults

def get_history_string():
    history_str = ""
    for entry in st.session_state.chat_history[-6:]:
        prefix = "User" if entry["role"] == "user" else "Assistant"
        history_str += f"{prefix}: {entry['content']}\n"
    return history_str.strip()

# =========================================================================
# PHASE 4: RETRIEVAL ENGINE — deterministic pandas execution.
# The LLM never computes numbers itself; it only produces the structured
# query above. This keeps every reported figure verifiably correct.
# =========================================================================

COMPARATORS = {
    "<": operator.lt, "<=": operator.le, ">": operator.gt,
    ">=": operator.ge, "==": operator.eq, "!=": operator.ne,
}

def filter_by_student(df, name):
    if not name:
        return df
    return df[df["Student_Name"].str.lower() == str(name).strip().lower()]

def apply_filter(df, column, op_str, value):
    if column not in df.columns or value is None:
        return None
    if column == "Assignment_3_Status":
        target = str(value).strip().lower()
        mask = df[column].astype(str).str.lower() == target
        if op_str == "!=":
            mask = ~mask
        return df[mask]
    series = pd.to_numeric(df[column], errors="coerce")
    try:
        num_val = float(value)
    except (TypeError, ValueError):
        return None
    op_func = COMPARATORS.get(op_str)
    if op_func is None:
        return None
    return df[op_func(series, num_val)]

def analyze_anomalies(subset):
    if subset is None or subset.empty:
        return None
    missing_a3 = len(subset[subset["Assignment_3_Status"] == "Missing"])
    if (missing_a3 / len(subset)) > 0.50:
        pct = int((missing_a3 / len(subset)) * 100)
        return f"{pct}% of this group has not submitted Assignment 3."
    return None

def run_retrieval(nlu, active_df):
    intent = nlu["intent"]
    column = nlu.get("column")
    student_name = nlu.get("student_name")

    base_df = active_df
    if student_name:
        base_df = filter_by_student(active_df, student_name)
        if base_df.empty:
            return {"type": "error", "detail": f"No student named '{student_name}' found in the current focus group."}

    if intent == "data_filter":
        result_df = apply_filter(base_df, column, nlu.get("filter_operator"), nlu.get("filter_value"))
        if result_df is None:
            return {"type": "error", "detail": "Could not identify which column or value to filter on."}
        return {"type": "filter", "df": result_df, "column": column}

    if intent == "data_aggregate":
        if column not in DATA_COLUMNS_NUMERIC:
            return {"type": "error", "detail": "That column isn't numeric, so it can't be averaged/summed."}
        series = pd.to_numeric(base_df[column], errors="coerce").dropna()
        if series.empty:
            return {"type": "error", "detail": "No numeric data available for that column in the current focus group."}
        func = nlu.get("aggregate_function") or "mean"
        value = getattr(series, func)()
        return {"type": "aggregate", "value": value, "column": column, "func": func, "n": len(series)}

    if intent == "data_lookup":
        cols = ["Student_Name", column] if column in base_df.columns else base_df.columns.tolist()
        return {"type": "lookup", "df": base_df[cols]}

    return {"type": "none"}

def summarize_retrieval(retrieval):
    t = retrieval.get("type")
    if t == "error":
        return f"ERROR: {retrieval['detail']}"
    if t == "aggregate":
        return f"{retrieval['func']}({retrieval['column']}) = {retrieval['value']:.2f}  (based on {retrieval['n']} students)"
    if t in ("filter", "lookup"):
        df = retrieval["df"]
        if df.empty:
            return "No matching students found."
        return df.to_csv(index=False)
    if t == "pattern":
        return retrieval["detail"]
    return "No structured data result for this turn."

# =========================================================================
# PHASE 5: GENERATION — phrase the final answer in the user's own language,
# grounded strictly in the retrieval result (never invents numbers).
# =========================================================================

def generate_grounded_response(user_query, language, retrieval):
    grounding = summarize_retrieval(retrieval)
    system = (
        f"You are a warm, concise assistant helping a teacher review student grades. "
        f"Reply ONLY in {lang_name(language)}. "
        f"Use the ground-truth result below as the sole source of any numbers or names — "
        f"never invent or alter a value. Keep it to 1-3 sentences unless listing several students. "
        f"If the result starts with 'ERROR:', apologize briefly and suggest how to rephrase, "
        f"still in {lang_name(language)}."
    )
    user_msg = f"Teacher's question: {user_query}\n\nGround-truth result:\n{grounding}"
    return call_llm(
        [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        temperature=0.3, max_tokens=350
    )

def generate_open_response(user_query, language):
    system = (
        f"You are a helpful, general-purpose assistant embedded inside a teacher's grading tool. "
        f"Reply ONLY in {lang_name(language)}, concisely and helpfully, on whatever the user asked, "
        f"even if unrelated to student data."
    )
    return call_llm(
        [{"role": "system", "content": system}, {"role": "user", "content": user_query}],
        temperature=0.5, max_tokens=400
    )

def generate_smalltalk(user_query, language):
    system = (
        f"You are a friendly grading assistant. The teacher just sent a greeting/small-talk message. "
        f"Reply briefly and warmly in {lang_name(language)}, and remind them (in one short clause) "
        f"they can ask about grades, averages, or patterns in either English or Arabic."
    )
    return call_llm(
        [{"role": "system", "content": system}, {"role": "user", "content": user_query}],
        temperature=0.4, max_tokens=150
    )

# =========================================================================
# PHASE 6: DIALOGUE POLICY — routes each turn to the right stage above and
# owns dialogue state (active cohort / focus group).
# =========================================================================

def handle_turn(user_input):
    history_str = get_history_string()
    nlu = run_nlu(user_input, history_str)
    language = nlu.get("language") or ("ar" if is_arabic(user_input) else "en")
    intent = nlu.get("intent")

    if intent == "smalltalk":
        return generate_smalltalk(user_input, language)

    if intent == "clarify":
        return nlu.get("clarification_question") or (
            "من فضلك وضّح أكثر." if language == "ar" else "Could you clarify that a bit more?"
        )

    if intent == "meta_clarify":
        last = next((e["content"] for e in reversed(st.session_state.chat_history) if e["role"] == "assistant"), None)
        if not last:
            return "لا توجد إجابة سابقة لأوضحها بعد." if language == "ar" else "I don't have a previous answer to clarify yet."
        system = f"Rephrase the following previous answer more simply and clearly, in {lang_name(language)}."
        return call_llm([{"role": "system", "content": system}, {"role": "user", "content": last}], temperature=0.3)

    if intent == "reset_focus":
        st.session_state.active_subset = None
        return "تم إعادة التركيز إلى الصف بالكامل." if language == "ar" else "Focus reset to the entire class."

    if intent == "general_knowledge":
        return generate_open_response(nlu.get("standalone_query_en", user_input), language)

    active_df = st.session_state.active_subset if st.session_state.active_subset is not None else st.session_state.df

    if intent == "pattern_analysis":
        insight = analyze_anomalies(active_df)
        retrieval = {"type": "pattern", "detail": insight or "No shared risk factor found in this group."}
        return generate_grounded_response(user_input, language, retrieval)

    retrieval = run_retrieval(nlu, active_df)

    if retrieval["type"] == "filter":
        st.session_state.active_subset = retrieval["df"].copy()

    return generate_grounded_response(user_input, language, retrieval)

# =========================================================================
# APPLICATION STATE + UI
# =========================================================================

if "df" not in st.session_state:
    st.session_state.df = ensure_synthetic_data()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_subset" not in st.session_state:
    st.session_state.active_subset = None

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
    st.header("🤖 Conversational Assistant / المساعد التحاوري")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if is_arabic(msg["content"]):
                st.markdown(f'<div dir="rtl" style="text-align:right">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.write(msg["content"])

    placeholder = "Ask about grades, averages, patterns — in English or Arabic / اسأل عن الدرجات أو المعدلات..."
    if user_input := st.chat_input(placeholder):
        with st.chat_message("user"):
            if is_arabic(user_input):
                st.markdown(f'<div dir="rtl" style="text-align:right">{user_input}</div>', unsafe_allow_html=True)
            else:
                st.write(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("Thinking… / لحظة من فضلك..."):
            response = handle_turn(user_input)

        with st.chat_message("assistant"):
            if is_arabic(response):
                st.markdown(f'<div dir="rtl" style="text-align:right">{response}</div>', unsafe_allow_html=True)
            else:
                st.write(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
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
