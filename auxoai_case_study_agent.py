

# ================================================================================

# Cell 0
!pip install groq sentence-transformers gradio --quiet

# ================================================================================

# Cell 1
import os
import json
import gradio as gr
from groq import Groq
from sentence_transformers import SentenceTransformer, util
import torch
from google.colab import userdata

# Pull key from Colab Secrets (no hardcoding needed)
client = Groq(api_key=userdata.get("GROQ_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Setup complete.")

# ================================================================================

# Cell 2
# All 9 case studies scraped from auxoai.com/case-studies
# Each entry: industry, problem, solution, key results

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
        "problem": "A global berry company undergoing ERP transformation faced slow, manual QA testing cycles that couldn't keep pace with their rollout timeline.",
        "solution": "AI augmented QA system that auto-generated and executed test cases against ERP workflows, reducing human testing burden.",
        "results": "50%+ reduction in test creation time, 75%+ first version accuracy.",
        "use_case_tag": "AINext / QA Automation"
    },
    {
        "title": "From Chatbot Pilot to Enterprise AI Backbone",
        "industry": "Sports Equipment / Golf",
        "problem": "A leading golf equipment brand had fragmented customer service systems across ERP, CRM, product specs, and R&D documents, causing slow resolution times and agent inefficiency.",
        "solution": "Enterprise AI platform (CHIP) built on Azure, Salesforce, and Oracle, unifying knowledge sources into a single agent layer for customer and employee queries.",
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

print(f"Loaded {len(case_studies)} case studies.")

# ================================================================================

# Cell 3
# Pre-compute embeddings for each case study
# We embed the combined problem + solution text for best semantic matching

def build_case_study_text(cs):
    return f"{cs['industry']}. Problem: {cs['problem']} Solution: {cs['solution']} Results: {cs['results']}"

corpus = [build_case_study_text(cs) for cs in case_studies]
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

print("Embeddings built for all case studies.")

# ================================================================================

# Cell 4
CONFIDENCE_THRESHOLD = 0.35  # Below this, escalate to human

def retrieve_top_match(prospect_input: str, top_k: int = 2):
    query_embedding = embedder.encode(prospect_input, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(scores, k=top_k)

    matches = []
    for score, idx in zip(top_results.values, top_results.indices):
        matches.append({
            "case_study": case_studies[idx.item()],
            "confidence": round(score.item(), 3)
        })
    return matches


def generate_pitch(prospect_input: str, match: dict) -> str:
    cs = match["case_study"]
    confidence = match["confidence"]

    system_prompt = """You are a forward deployed AI expert at AuxoAI.
Your job is to write a sharp, one-paragraph pitch that connects a prospect's specific problem
to the most relevant AuxoAI case study.
Be specific, reference real results, and make the connection feel tailored, not generic.
Write in a confident, consultative tone. No bullet points. 3-5 sentences max."""

    user_prompt = f"""Prospect problem: {prospect_input}

Most relevant AuxoAI case study:
- Title: {cs['title']}
- Industry: {cs['industry']}
- Problem solved: {cs['problem']}
- Solution: {cs['solution']}
- Results: {cs['results']}

Write a tailored one-paragraph pitch connecting AuxoAI's experience to this prospect's problem."""

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


def run_agent(prospect_input: str):
    if not prospect_input.strip():
        return "Please describe the prospect's problem.", "", ""

    matches = retrieve_top_match(prospect_input, top_k=2)
    top_match = matches[0]
    confidence = top_match["confidence"]
    cs = top_match["case_study"]

    # Low confidence: escalate
    if confidence < CONFIDENCE_THRESHOLD:
        result = (
            f"LOW CONFIDENCE MATCH (score: {confidence})\n\n"
            "No strong case study match found for this problem.\n"
            "Recommended action: Escalate to a human AuxoAI expert for a custom scoping call.\n\n"
            f"Closest match attempted: {cs['title']} ({cs['industry']})"
        )
        return result, f"{confidence} - Below threshold ({CONFIDENCE_THRESHOLD})", "ESCALATE TO HUMAN"

    # High confidence: generate pitch
    pitch = generate_pitch(prospect_input, top_match)

    confidence_label = (
        "HIGH" if confidence >= 0.55
        else "MEDIUM" if confidence >= 0.42
        else "LOW-MODERATE"
    )

    match_summary = (
        f"Case Study: {cs['title']}\n"
        f"Industry: {cs['industry']}\n"
        f"Use Case Tag: {cs['use_case_tag']}\n"
        f"Key Results: {cs['results']}"
    )

    return pitch, f"{confidence} ({confidence_label})", match_summary

# ================================================================================

# Cell 5
with gr.Blocks(theme=gr.themes.Soft(), title="AuxoAI Case Study Discovery Agent") as demo:

    gr.Markdown("""
    # AuxoAI Case Study Discovery Agent
    **Built by Abhishek Singh | Forward Deployed AI PM Portfolio Project**

    Describe a prospect's business problem. The agent retrieves the most relevant AuxoAI case study
    and generates a tailored pitch in seconds. Low-confidence matches are flagged for human escalation.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            prospect_input = gr.Textbox(
                label="Prospect Problem Description",
                placeholder="e.g. We are a mid-size pharma company. Our SDR team is overwhelmed and we are missing high-value leads because we don't have a good way to score or prioritize our pipeline.",
                lines=5
            )
            submit_btn = gr.Button("Run Agent", variant="primary")

        with gr.Column(scale=1):
            confidence_output = gr.Textbox(label="Confidence Score", interactive=False)
            match_output = gr.Textbox(label="Best Matching Case Study", lines=5, interactive=False)

    pitch_output = gr.Textbox(label="Generated Pitch", lines=6, interactive=False)

    gr.Markdown("""
    ---
    **How it works:** Sentence embeddings match the prospect's problem to AuxoAI's case study knowledge base.
    Scores above 0.35 generate an LLM-powered pitch. Below 0.35, the agent escalates to a human expert.
    """)

    submit_btn.click(
        fn=run_agent,
        inputs=[prospect_input],
        outputs=[pitch_output, confidence_output, match_output]
    )

    gr.Examples(
        examples=[
            ["We are a healthcare SaaS company. Our sales team spends too much time chasing cold leads and our win rates are declining. We need a smarter way to prioritize which accounts to go after."],
            ["We are a national retailer with 10 years of sales data sitting in a legacy data warehouse. We want to modernize our data infrastructure but cannot afford any downtime during migration."],
            ["We run a telecom customer retention center. Our agents take weeks to train and our save rates are too low. We need to get agents productive faster."],
            ["We are a B2B manufacturing company struggling to scale our outbound prospecting. We want to reach more potential customers without hiring more sales reps."],
        ],
        inputs=[prospect_input]
    )

demo.launch(debug=True)

# ================================================================================

# Cell 6
import json
from google.colab import _message

# Get the notebook JSON
nb = _message.blocking_request('get_ipynb')['ipynb']

# Extract code cells
all_code = "\n\n# " + "="*80 + "\n\n"
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        all_code += f"# Cell {i}\n"
        all_code += ''.join(cell['source'])
        all_code += "\n\n# " + "="*80 + "\n\n"

# Save to a .py file
with open('entire_notebook.py', 'w') as f:
    f.write(all_code)

# Print all code (optional)
print(all_code)

