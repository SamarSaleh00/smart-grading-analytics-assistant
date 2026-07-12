# 🤖 AI Educational Assistant, Smart Grading Analytics

A conversational, multi-turn analytics assistant for teachers, built with **Streamlit**, **TAPAS** (table question-answering), and **FLAN-T5** (query rewriting). Ask questions about student grades in natural language, isolate cohorts, and get proactive insights  all through a chat interface with a live-updating data dashboard.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Features

- **Conversational grade queries** : ask things like *"Give me a list of students who scored below 60% on the final exam"* in plain English.
- **Multi-turn context tracking** : follow-up questions like *"What is their average score in the assignments?"* correctly resolve pronouns (`their`, `them`, `it`) back to the previously mentioned cohort, using a FLAN-T5-powered query rewriter.
- **Stateful cohort isolation** : once a query returns a subset of students, the dashboard "locks" onto that group so subsequent questions and calculations apply only to them, until reset.
- **Anomaly / pattern detection** : asking about "patterns" or "common" issues automatically checks the active cohort for shared risk factors (e.g. missing assignments) and proactively suggests next steps.
- **Clarification guardrails** : ambiguous questions (e.g. "quiz score" without specifying which quiz) trigger a clarifying question instead of guessing.
- **Live split-screen UI** : a real-time student data table sits alongside the chat, always reflecting the current focus cohort.

## 🧠 How it works

| Layer | Component | Purpose |
|---|---|---|
| Data | Synthetic pandas dataset (`st.cache_data`) | Simulates a class roster with quiz, assignment, and exam scores |
| Query rewriting | `google/flan-t5-small` | Rewrites follow-up questions into standalone queries by resolving pronouns from chat history |
| Table QA | `google/tapas-base-finetuned-wtq` | Answers natural-language questions directly against the student table |
| Dialogue policy | Custom routing logic | Decides whether to answer via pattern analysis, aggregate math, or table QA, and manages the active cohort in `st.session_state` |
| Guardrails | `check_clarification_gate()` | Detects ambiguous queries and asks for clarification before answering |

Models are loaded once and cached across reruns via `st.cache_resource`, so the app stays responsive after the first load.

## 📂 Project structure

```
.
├── app.py              # Streamlit application (UI + orchestration logic)
├── requirements.txt    # Python dependencies
└── README.md
```

## 🚀 Getting started

### 1. Clone the repo

```bash
git clone https://github.com/SamarSaleh00/smart-grading-analytics-assistant.git
cd smart-grading-analytics-assistant
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

> **Note:** The first run will download the FLAN-T5 and TAPAS models from Hugging Face (a few hundred MB total), so it may take a minute or two depending on your connection.

## 💬 Example conversation

```
Teacher: Give me a list of students who scored below 60% on the final exam.
Assistant: Alex, Omar, Sarah

Teacher: What is their average score in the assignments?
Assistant: The average assignment score for this group is 57.5% (A1: 62.0%, A2: 53.0%).

Teacher: Is there a common pattern?
Assistant: Yes, 100% of these students have not submitted Assignment 3.
           Would you like me to draft a reminder email for them?
```

## 🛠️ Tech stack

- [Streamlit](https://streamlit.io/) : UI and app state management
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) : TAPAS and FLAN-T5 models
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) : data handling
- [PyTorch](https://pytorch.org/) : model backend

## 🗺️ Possible next steps

- Swap the synthetic dataset for a CSV/Google Sheets upload
- Add authentication so multiple teachers can use isolated sessions
- Extend the dialogue policy with more analytic intents (e.g. trend-over-time, grade distributions)
- Add an "export cohort" or "send reminder email" action tied to the anomaly detector

## 📄 License

This project is licensed under the MIT License, feel free to use and adapt it.
