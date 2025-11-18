# ===============================================================
#  app.py — COMPLETE NEW VERSION (Stable, Clean, and Fully Working)
# ===============================================================
from dotenv import load_dotenv
load_dotenv()

import os
import re
import streamlit as st
import PyPDF2
from io import StringIO
from crewai import Agent, Task, Crew, Process, LLM

# --------------------------------------------------------------
# Streamlit Setup
# --------------------------------------------------------------
st.set_page_config(page_title="🧠 Multi-Agent Meeting Prep", layout="wide")

# Dark/Light Theme Toggle
theme = st.sidebar.radio("Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
        <style>
        .stApp { background-color:#111; color:white; }
        textarea, input { background-color:#222 !important; color:white !important; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp { background-color:white; color:black; }
        textarea, input { background-color:#f9f9f9 !important; color:black !important; }
        </style>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
def extract_pdf_text(file):
    """Extract readable text from user-uploaded PDF."""
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text


def parse_final_output(output: str):
    """Extract required sections from the Crew final output."""

    sections = {
        "SUMMARY": "",
        "KEY DISCUSSION POINTS": "",
        "ACTION ITEMS": "",
        "SUGGESTED QUESTIONS": "",
        "RISKS AND MISSING INFO": "",
        "MEETING SCRIPT": ""
    }

    pattern = r"#\s*(SUMMARY|KEY DISCUSSION POINTS|ACTION ITEMS|SUGGESTED QUESTIONS|RISKS AND MISSING INFO|MEETING SCRIPT)\s*(.*?)(?=\n#|$)"
    matches = re.findall(pattern, output, re.DOTALL)

    for header, content in matches:
        sections[header] = content.strip()

    return sections


# --------------------------------------------------------------
# CrewAI Pipeline
# --------------------------------------------------------------
def run_meeting_prep_crew(full_text: str):

    if not os.environ.get("GEMINI_API_KEY"):
        return "ERROR: GEMINI_API_KEY missing in environment."

    gemini_llm = LLM(model="gemini/gemini-2.5-flash", temperature=0.1)

    text = full_text[:100000]  # Safety limit

    # Agents
    summarizer = Agent(
        role="Document Summarizer",
        goal="Summarize documents clearly.",
        backstory="Expert in condensing long text.",
        llm=gemini_llm
    )

    points = Agent(
        role="Discussion Point Extractor",
        goal="Extract key points.",
        backstory="Senior analyst.",
        llm=gemini_llm
    )

    actions = Agent(
        role="Action Item Extractor",
        goal="Find all tasks.",
        backstory="Project manager.",
        llm=gemini_llm
    )

    questions = Agent(
        role="Question Generator",
        goal="Generate 5 critical questions.",
        backstory="Advisor.",
        llm=gemini_llm
    )

    risks = Agent(
        role="Risk Analyst",
        goal="Find risks and missing info.",
        backstory="Risk specialist.",
        llm=gemini_llm
    )

    script = Agent(
        role="Report Compiler",
        goal="Create final structured meeting prep report.",
        backstory="Lead coordinator.",
        llm=gemini_llm
    )

    # Tasks
    t1 = Task(
        description=f"Summarize:\n{text}",
        expected_output="2–3 paragraphs.",
        agent=summarizer
    )

    t2 = Task(
        description="List 3–5 strategic discussion points.",
        expected_output="Bulleted list.",
        agent=points
    )

    t3 = Task(
        description="Extract ALL action items.",
        expected_output="Action list.",
        agent=actions
    )

    t4 = Task(
        description="Generate exactly 5 questions.",
        expected_output="List of 5 questions.",
        agent=questions
    )

    t5 = Task(
        description="Identify risks and missing information.",
        expected_output="Paragraph.",
        agent=risks
    )

    t6 = Task(
        description="""
Use all previous results and produce the final structured output:

# SUMMARY
<summary>

# KEY DISCUSSION POINTS
<points>

# ACTION ITEMS
<items>

# SUGGESTED QUESTIONS
<questions>

# RISKS AND MISSING INFO
<risks>

# MEETING SCRIPT
<script>
        """,
        expected_output="Formatted final output",
        agent=script
    )

    crew = Crew(
        agents=[summarizer, points, actions, questions, risks, script],
        tasks=[t1, t2, t3, t4, t5, t6],
        process=Process.sequential,
        verbose=False
    )

    # Always return string
    result = crew.kickoff()

    # FIXED: handle every possible Crew output structure
    if hasattr(result, "raw_output") and result.raw_output:
        return result.raw_output

    if hasattr(result, "final_output") and result.final_output:
        return result.final_output

    if hasattr(result, "output") and result.output:
        return result.output

    return str(result)


# --------------------------------------------------------------
# Main UI
# --------------------------------------------------------------
st.title("🧠 Multi-Agent AI Meeting Prep Agent")
st.write("Upload one or more documents and generate a complete meeting preparation package.")

uploaded = st.file_uploader("Upload files (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)

full_text = ""

if uploaded:
    for f in uploaded:
        if f.type == "application/pdf":
            full_text += extract_pdf_text(f) + "\n"
        else:
            full_text += f.getvalue().decode("utf-8") + "\n"


if st.button("Run Agents"):
    if not full_text.strip():
        st.warning("Please upload at least 1 document.")
        st.stop()

    with st.spinner("Running AI Agents... This may take 5–15 seconds..."):
        final_output = run_meeting_prep_crew(full_text)

    st.session_state["raw_output"] = final_output
    st.success("Processing complete!")


# --------------------------------------------------------------
# Output display
# --------------------------------------------------------------
if "raw_output" in st.session_state:

    raw = st.session_state["raw_output"]

    with st.expander("🐞 RAW OUTPUT (Click to expand)"):
        st.code(raw)

    parsed = parse_final_output(raw)

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📄 Summary",
        "💡 Key Points",
        "✅ Action Items",
        "❓ Questions",
        "⚠️ Risks",
        "🗣️ Script"
    ])

    with t1: st.text_area("Summary", parsed["SUMMARY"], height=300)
    with t2: st.text_area("Key Discussion Points", parsed["KEY DISCUSSION POINTS"], height=300)
    with t3: st.text_area("Action Items", parsed["ACTION ITEMS"], height=300)
    with t4: st.text_area("Suggested Questions", parsed["SUGGESTED QUESTIONS"], height=300)
    with t5: st.text_area("Risks & Missing Info", parsed["RISKS AND MISSING INFO"], height=300)
    with t6: st.text_area("Meeting Script", parsed["MEETING SCRIPT"], height=400)
