import os
import torch
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AuxoAI Case Study Discovery Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Load CSS ─────────────────────────────────────────────────
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Init clients ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return client, embedder

client, embedder = load_models()

# ── Knowledge base ───────────────────────────────────────────
case_studies = [
    {
        "title": "Predictive Growth Through AI Powered Lead Scoring",
        "industry": "Healthcare / EHR",
        "problem": "A leading EHR enterprise needed to improve sales win rates by identifying and prioritizing the highest value leads from a large prospect pool.",
        "solution": "AI powered lead scoring model that ranked prospects by conversion likelihood, enabling sales reps to focus on highest intent accounts.",
        "results": "6x higher win rates, 58% of wins captured from top scored leads.",
        "use_case_tag": "MarketNext / Lead Scoring"
    },
    {
        "title": "Turning Data Chaos into Intelligent Sales Decisions",
        "industry": "CPG / Consumer Products",
        "problem": "A Fortune 500 consumer products group had fragmented data across systems, leading to manual reporting bottlenecks and inconsistent sales intelligence.",
        "solution": "AI powered analytics platform creating a single source of truth, automating reporting and surfacing actionable sales insights.",
        "results": "75% reduction in manual reporting, unified data foundation across teams.",
        "use_case_tag": "DataNext / Sales Analytics"
    },
    {
        "title": "Scaling ERP Testing with AI Augmented QA",
        "industry": "Food and Agriculture",
        "problem": "A global berry company undergoing ERP transformation faced slow, manual QA testing cycles that could not keep pace with their rollout timeline.",
        "solution": "AI augmented QA system that auto-generated and executed test cases against ERP workflows, reducing human testing burden.",
        "results": "50%+ reduction in test creation time, 75%+ first version accuracy.",
        "use_case_tag": "AINext / QA Automation"
    },
    {
        "title": "From Chatbot Pilot to Enterprise AI Backbone",
        "industry": "Sports Equipment / Golf",
        "problem": "A leading golf equipment brand had fragmented customer service systems across ERP, CRM, product specs, and R&D documents, causing slow resolution times and agent inefficiency.",
        "solution": "Enterprise AI platform built on Azure, Salesforce, and Oracle, unifying knowledge sources into a single agent layer for customer and employee queries.",
        "results": "Thousands of interactions per month, 5 global regions in 6 months, expanded to R&D, Operations, HR, and Finance.",
        "use_case_tag": "AINext / Enterprise AI Platform"
    },
    {
        "title": "Modernizing Decades of Legacy Data with Zero Downtime",
        "industry": "CPG / Food Manufacturing",
        "problem": "North America's leading food manufacturer had a multi-terabyte legacy data warehouse that needed migration without disrupting operations.",
        "solution": "AI assisted data modernization pipeline enabling phased migration with validation checkpoints and zero downtime delivery.",
        "results": "60% faster migration, 33% lower cost.",
        "use_case_tag": "DataNext / Data Modernization"
    },
    {
        "title": "Building Enterprise AI Strategy Across Four Core Functions",
        "industry": "Food and Agriculture",
        "problem": "The world's largest berry producer needed to identify where AI could deliver the highest ROI across a complex enterprise with multiple business functions.",
        "solution": "Structured AI opportunity assessment identifying 90+ use cases, prioritizing 3 for immediate execution across business and IT alignment.",
        "results": "90+ AI opportunities identified, 3 priority use cases ready for execution, full business-IT alignment achieved.",
        "use_case_tag": "AINext / AI Strategy"
    },
    {
        "title": "Accelerating Sales Pipeline with AI Powered Prospecting",
        "industry": "Environmental Services / Waste Management",
        "problem": "A leading sustainable waste solutions provider needed to scale lead generation across diverse industrial verticals without adding SDR headcount.",
        "solution": "AI-SDR platform that profiled, scored, and engaged prospects with personalized outreach, running geo-specific campaigns autonomously.",
        "results": "2x-5x higher engagement, 10x scalability, went live in 1 week.",
        "use_case_tag": "MarketNext / AI-SDR"
    },
    {
        "title": "Transforming Lead Prioritization with ML Powered Intelligence",
        "industry": "Healthcare / EHR",
        "problem": "A leading EHR company needed a smarter way to prioritize leads so sales reps could focus effort on accounts most likely to convert.",
        "solution": "ML powered lead scoring model integrated into existing sales workflows, surfacing intent signals and ranking accounts by conversion probability.",
        "results": "2x higher win rates from intelligently prioritized leads.",
        "use_case_tag": "MarketNext / Lead Scoring"
    },
    {
        "title": "Transforming Agent Readiness with AI Powered Retention Training",
        "industry": "Telecom",
        "problem": "A national telecom provider needed to upskill customer retention agents faster, as long training cycles were delaying frontline readiness and impacting save rates.",
        "solution": "AI driven training platform delivering personalized learning paths and real time coaching for retention agents.",
        "results": "20% improvement in proficiency, 5% improvement in save rates, significant reduction in training time.",
        "use_case_tag": "AINext / Agent Training"
    }
]

@st.cache_resource
def build_embeddings():
    corpus = [
        f"{cs['industry']}. Problem: {cs['problem']} Solution: {cs['solution']} Results: {cs['results']}"
        for cs in case_studies
    ]
    return embedder.encode(corpus, convert_to_tensor=True)

corpus_embeddings = build_embeddings()

CONFIDENCE_THRESHOLD = 0.35

# ── Core logic ───────────────────────────────────────────────
def retrieve_top_match(prospect_input: str):
    query_embedding = embedder.encode(prospect_input, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(scores, k=2)
    return [
        {"case_study": case_studies[idx.item()], "confidence": round(score.item(), 3)}
        for score, idx in zip(top_results.values, top_results.indices)
    ]

def generate_pitch(prospect_input: str, match: dict) -> str:
    cs = match["case_study"]
    system_prompt = """You are a forward deployed AI expert at AuxoAI.
Write a sharp, one-paragraph pitch connecting a prospect's specific problem to the most relevant AuxoAI case study.
Be specific, reference real results, and make the connection feel tailored, not generic.
Confident, consultative tone. No bullet points. 3-5 sentences."""

    user_prompt = f"""Prospect problem: {prospect_input}

Most relevant AuxoAI case study:
- Title: {cs['title']}
- Industry: {cs['industry']}
- Problem solved: {cs['problem']}
- Solution: {cs['solution']}
- Results: {cs['results']}

Write the pitch."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=300,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

def confidence_label(score: float) -> tuple:
    if score >= 0.55:
        return "HIGH", "confidence-high"
    elif score >= 0.42:
        return "MEDIUM", "confidence-medium"
    elif score >= 0.35:
        return "LOW-MODERATE", "confidence-low"
    else:
        return "BELOW THRESHOLD", "confidence-escalate"

# ── UI ───────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AuxoAI Internal Tool</div>
    <h1 class="hero-title">Case Study <span>Discovery Agent</span></h1>
    <p class="hero-sub">Describe a prospect's problem. Get the right case study and a tailored pitch in seconds.</p>
</div>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────
if "prospect_text" not in st.session_state:
    st.session_state.prospect_text = ""

examples = {
    "Healthcare SDR":        "We are a healthcare SaaS company selling EHR software. Our win rates are dropping and reps waste time on low-intent accounts. We need smarter lead prioritization.",
    "B2B Logistics":         "We are a B2B logistics company. Our two SDRs can only reach so many prospects weekly. We need to scale pipeline without adding headcount.",
    "Food Mfg Data":         "We are a food manufacturer. Data is spread across 5 systems and analysts spend all day pulling manual reports. We need a single source of truth.",
    "Telecom Training":      "We run a telecom retention center. New agents take 6 weeks to ramp and save rates are inconsistent. We need faster training.",
    "Law Firm (escalation)": "We are a law firm looking to automate contract review and reduce due diligence time."
}

# ── Main columns ─────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown('<p class="card-label">Prospect Problem Description</p>', unsafe_allow_html=True)
    st.markdown('<p class="example-label">Try an example:</p>', unsafe_allow_html=True)

    btn_cols = st.columns(3)
    for i, (label, text) in enumerate(examples.items()):
        with btn_cols[i % 3]:
            if st.button(label, key=f"ex_{i}", use_container_width=True):
                st.session_state.prospect_text = text
                st.rerun()

    prospect_input = st.text_area(
        label="",
        value=st.session_state.prospect_text,
        placeholder="e.g. We are a mid-size pharma company. Our SDR team is overwhelmed and we are missing high-value leads because we have no way to score or prioritize our pipeline.",
        height=160,
        label_visibility="collapsed"
    )

    run = st.button("Run Agent", type="primary", use_container_width=True)

with col_right:
    st.markdown('<p class="card-label">How It Works</p>', unsafe_allow_html=True)
    st.markdown("""
<div class="how-it-works">
    <div class="step"><span class="step-num">1</span><span>Your input is converted to a semantic vector</span></div>
    <div class="step"><span class="step-num">2</span><span>Scored against 9 AuxoAI case studies</span></div>
    <div class="step"><span class="step-num">3</span><span>Confidence check at 0.35 threshold</span></div>
    <div class="step"><span class="step-num">4</span><span>Groq generates a tailored pitch, or escalates to human</span></div>
</div>
""", unsafe_allow_html=True)

# ── Results ──────────────────────────────────────────────────
if run and prospect_input.strip():
    with st.spinner("Retrieving best match and generating pitch..."):
        matches = retrieve_top_match(prospect_input)
        top = matches[0]
        score = top["confidence"]
        cs = top["case_study"]
        label, css_class = confidence_label(score)

        st.markdown("---")

        if score < CONFIDENCE_THRESHOLD:
            st.markdown(f"""
<div class="result-card escalate">
    <div class="result-header">
        <span class="badge {css_class}">ESCALATE TO HUMAN</span>
        <span class="score-display">Score: {score}</span>
    </div>
    <p class="escalate-msg">No strong case study match found for this problem.</p>
    <p class="escalate-sub">Recommended action: Route to a human AuxoAI expert for a custom scoping call.</p>
    <p class="escalate-attempt">Closest attempted match: <strong>{cs['title']}</strong> ({cs['industry']})</p>
</div>
""", unsafe_allow_html=True)

        else:
            pitch = generate_pitch(prospect_input, top)

            st.markdown(f"""
<div class="result-header">
    <span class="badge {css_class}">{label} CONFIDENCE</span>
    <span class="score-display">Score: {score}</span>
</div>
""", unsafe_allow_html=True)

            r1, r2 = st.columns([3, 2], gap="large")
            with r1:
                st.markdown('<p class="card-label">Generated Pitch</p>', unsafe_allow_html=True)
                st.markdown(f'<div class="pitch-box">{pitch}</div>', unsafe_allow_html=True)
            with r2:
                st.markdown('<p class="card-label">Matched Case Study</p>', unsafe_allow_html=True)
                st.markdown(f"""
<div class="match-box">
    <p class="match-title">{cs['title']}</p>
    <div class="match-meta">
        <span class="meta-tag">{cs['industry']}</span>
        <span class="meta-tag">{cs['use_case_tag']}</span>
    </div>
    <p class="match-results">{cs['results']}</p>
</div>
""", unsafe_allow_html=True)

elif run and not prospect_input.strip():
    st.warning("Please describe the prospect's problem before running the agent.")

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built by <a href="https://linkedin.com/in/aabhishek-singh/" target="_blank">Abhishek Singh</a> &nbsp;|&nbsp;
    <a href="https://github.com/codegeekAbhi/" target="_blank">GitHub</a> &nbsp;|&nbsp;
    <a href="https://www.notion.so/Hi-I-m-Abhishek-Singh-21b6d321e30b804eab8ad37f2783be09" target="_blank">Portfolio</a>
</div>
""", unsafe_allow_html=True)
