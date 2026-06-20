# AuxoAI Case Study Discovery Agent

**Built by Abhishek Singh | Forward Deployed AI PM Portfolio Project**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B)](https://your-streamlit-link-here)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Llama%203.3--70b-orange)](https://groq.com)

---

## What This Is

A forward deployment simulation tool built specifically for AuxoAI's go-to-market workflow.

A sales rep or forward deployed expert walks into a prospect conversation. The prospect describes their problem. Instead of manually searching through case studies, this agent retrieves the most relevant AuxoAI case study using semantic similarity and generates a tailored one-paragraph pitch in seconds.

Low-confidence matches are flagged for human escalation rather than generating a hallucinated pitch. The agent knows what it does not know.

---

## Forward Deployment Brief

### Client Problem Statement

Enterprise sales teams at consulting and AI services firms waste significant time manually matching prospect problems to past work. When a prospect says "we have fragmented data and our analysts spend all day pulling reports," the rep needs to instantly recall the right case study, connect it to the prospect's specific context, and frame it as a pitch, all within a live conversation. This cognitive load slows deal velocity and leads to generic, unconvincing pitches.

### Why This Agent, Why Now

AuxoAI's differentiation is speed: strategy to deployed AI in weeks, not quarters. That same speed standard should apply to how AuxoAI sells. A forward deployed AI PM's first job at any client is to identify the highest-leverage intervention point and build something fast. This agent demonstrates exactly that instinct applied to AuxoAI's own GTM motion.

The Google Cloud Gemini Enterprise Business Unit partnership (April 2026) means AuxoAI's case study library will grow rapidly across new verticals. A retrieval agent that scales with that library is a natural internal tool.

### Architecture in Plain English
Prospect types a problem description
Sentence embedding model converts it to a vector
(all-MiniLM-L6-v2, runs locally, no API cost)
Cosine similarity scored against all 9 AuxoAI case studies
Confidence check:
Score >= 0.35: retrieve top match, send to Groq (Llama 3.3-70b)
Score < 0.35:  skip LLM, return ESCALATE TO HUMAN flag
Groq generates a tailored one-paragraph pitch
referencing real AuxoAI results and the prospect's specific context
Output: Pitch + Confidence Score + Matched Case Study
No vector database. No paid embedding API. Runs entirely on Groq's free tier and local sentence transformers.

### What Week Two Looks Like

- Connect to Salesforce via API: retrieved case study and generated pitch auto-populate into the opportunity notes field when a rep opens a prospect record
- Add a case study ingestion pipeline: upload a new PDF or URL, agent automatically embeds and indexes it into the knowledge base without touching the code
- Expand to AuxoAI's whitepaper and podcast library for deeper retrieval context
- Add multi-match output: show top 2 case studies when confidence scores are close, letting the rep choose

### The One Risk and How It Is Handled

**Risk: Hallucination on weak matches**

If the prospect's problem is outside AuxoAI's existing case study coverage (e.g. legal tech, contract review, niche manufacturing), a naive LLM would still generate a pitch, just a vague or fabricated one. That erodes trust in a live sales call.

**Mitigation: Confidence threshold guardrail**

The agent sets a hard threshold at cosine similarity score 0.35. Below that, it returns an explicit escalation flag instead of generating a pitch. The output reads: "No strong case study match found. Recommended action: Escalate to a human AuxoAI expert for a custom scoping call." The agent is designed to be honest about the boundaries of its knowledge.

---

## Tech Stack

| Component | Tool |
|-----------|------|
| LLM | Groq API, Llama 3.3-70b-versatile |
| Embeddings | sentence-transformers, all-MiniLM-L6-v2 |
| Similarity | PyTorch cosine similarity |
| UI | Streamlit |
| Knowledge Base | 9 AuxoAI case studies, hand-curated from auxoai.com |

---

## Run Locally

```bash
git clone https://github.com/codegeekAbhi/auxoai-discovery-agent
cd auxoai-discovery-agent
pip install -r requirements.txt
streamlit run app.py
```

Add your Groq API key to a `.env` file:

---

## About the Builder

Abhishek Singh is an MBA graduate from UC Davis (June 2026) with 8 years across Amazon, Deloitte, TCS, and BLCK UNICRN in product management, data engineering, and consulting.

[LinkedIn](https://linkedin.com/in/aabhishek-singh/) 
[GitHub](https://github.com/codegeekAbhi/) 
[Portfolio](https://www.notion.so/Hi-I-m-Abhishek-Singh-21b6d321e30b804eab8ad37f2783be09)
