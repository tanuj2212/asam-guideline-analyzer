# -------------------------------------------------------------------
# Living Guideline Pipeline - Streamlit App
# -------------------------------------------------------------------

import streamlit as st
import os
import json
import re
import time
import math
import requests
from io import BytesIO
from datetime import datetime
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher, unified_diff

# embeddings / retrieval
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# PDF
from PyPDF2 import PdfReader

# Set page config
st.set_page_config(
    page_title="ASAM Guideline Analyzer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# CONFIG
# -------------------------
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Updated PDF URLs using reliable sources
ASAM_2015_PDF = "https://www.asam.org/docs/default-source/quality-science/npg-jam-supplement.pdf"
ASAM_2020_PDF = "https://www.asam.org/docs/default-source/guidelines/npg-jam-supplement.pdf"

# Fallback URLs if primary ones fail
FALLBACK_2015_PDF = "https://web.archive.org/web/20220407175334if_/https://www.asam.org/docs/default-source/quality-science/npg-jam-supplement.pdf"
FALLBACK_2020_PDF = "https://web.archive.org/web/20220407175414if_/https://www.asam.org/docs/default-source/guidelines/npg-jam-supplement.pdf"

OUTDIR = "living_guideline_run"
os.makedirs(OUTDIR, exist_ok=True)

NUM_PAPERS = 10
TOP_K = 5
EMBED_MODEL = "all-MiniLM-L6-v2"
OPENAI_CHAT_MODEL = "gpt-4o-mini"

# -------------------------
# UTIL: PubMed search (2015-2019)
# -------------------------
def pubmed_search_ids(query: str, retmax: int = 10, mindate="2015", maxdate="2019"):
    """Return list of PMIDs for query constrained to date range (publication date)."""
    # PubMed date range filter uses [PDAT] e.g., 2015:2019[PDAT]
    dated_query = f"{query} AND {mindate}:{maxdate}[PDAT]"
    params = {"db": "pubmed", "term": dated_query, "retmax": retmax, "retmode": "json"}
    try:
        r = requests.get(f"{PUBMED_BASE}/esearch.fcgi", params=params, timeout=30)
        r.raise_for_status()
        res = r.json()
        idlist = res.get("esearchresult", {}).get("idlist", [])
        return idlist
    except Exception as e:
        st.error(f"PubMed search error: {e}")
        return []

def pubmed_fetch_abstracts(id_list):
    if not id_list:
        return []
    try:
        ids = ",".join(id_list)
        params = {"db": "pubmed", "id": ids, "retmode": "xml"}
        r = requests.get(f"{PUBMED_BASE}/efetch.fcgi", params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        records = []
        for art in root.findall(".//PubmedArticle"):
            try:
                pmid = art.findtext(".//PMID")
                title = art.findtext(".//ArticleTitle") or ""
                abs_nodes = art.findall(".//AbstractText")
                abstract = " ".join([ET.tostring(n, encoding='unicode', method='text').strip() for n in abs_nodes]) if abs_nodes else ""
                journal = art.findtext(".//Journal/Title") or ""
                year = art.findtext(".//Journal/JournalIssue/PubDate/Year") or ""
                records.append({"pmid": pmid, "title": title, "abstract": abstract, "journal": journal, "year": year})
            except Exception:
                continue
        return records
    except Exception as e:
        st.error(f"PubMed fetch error: {e}")
        return []

# -------------------------
# UTIL: download & extract PDF text (ROBUST VERSION)
# -------------------------
def download_pdf_text(url, fallback_url=None, max_retries=2):
    """Download PDF with retry logic and fallback URLs."""
    urls_to_try = [url]
    if fallback_url:
        urls_to_try.append(fallback_url)

    for current_url in urls_to_try:
        for attempt in range(max_retries):
            try:
                st.info(f"Downloading PDF from {current_url} (attempt {attempt + 1})")
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                r = requests.get(current_url, timeout=30, headers=headers)
                r.raise_for_status()

                # Validate PDF content
                if not r.content.startswith(b'%PDF'):
                    st.warning("Response doesn't appear to be a valid PDF")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    else:
                        raise ValueError("Invalid PDF content")

                reader = PdfReader(BytesIO(r.content))
                pages = []
                for p in reader.pages:
                    try:
                        text = p.extract_text() or ""
                        pages.append(text)
                    except Exception as e:
                        st.warning(f"Could not extract text from page: {e}")
                        pages.append("")

                st.success(f"Successfully downloaded PDF with {len(pages)} pages")
                return "\n\n".join(pages)

            except requests.exceptions.RequestException as e:
                st.error(f"Download failed: {e}")
                if attempt < max_retries - 1:
                    st.info("Retrying...")
                    time.sleep(2)
                elif current_url != urls_to_try[-1]:
                    st.info("Trying fallback URL...")
                    break  # Break to next URL
                else:
                    raise Exception(f"Failed to download PDF from all URLs: {e}")

    raise Exception("All download attempts failed")

def extract_excerpts_by_keyword(pdf_text: str, keyword: str, window=1000, max_occurrences=5):
    """Return a list of excerpt strings around keyword occurrences (case-insensitive)."""
    if not pdf_text or not keyword:
        return []

    lc = pdf_text.lower()
    kw = keyword.lower()
    excerpts = []
    start = 0

    while True:
        idx = lc.find(kw, start)
        if idx == -1:
            break
        a = max(0, idx - window//2)
        b = min(len(pdf_text), idx + len(kw) + window//2)
        excerpt = pdf_text[a:b].strip()
        if excerpt and excerpt not in excerpts:
            excerpts.append(excerpt)
        start = idx + len(kw)
        if len(excerpts) >= max_occurrences:
            break

    return excerpts

# -------------------------
# UTIL: embedding + FAISS
# -------------------------
def build_embeddings_index(passages, model_name=EMBED_MODEL):
    """Build FAISS index from text passages."""
    if not passages:
        raise ValueError("No passages provided for embedding")

    model = SentenceTransformer(model_name)
    embs = model.encode(passages, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    return index, embs, model

def retrieve_topk(index, model, question, passages, meta, k=TOP_K):
    """Retrieve top-k most relevant passages for a question."""
    if not passages:
        return []

    q_emb = model.encode([question], convert_to_numpy=True).astype("float32")
    k = min(k, len(passages))  # Ensure k doesn't exceed number of passages
    D, I = index.search(q_emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < len(passages):  # Safety check
            results.append({"passage": passages[idx], "meta": meta[idx], "score": float(score)})
    return results

# -------------------------
# UTIL: LLM call (OpenAI ChatCompletion)
# -------------------------
def call_openai_chat(prompt, model=OPENAI_CHAT_MODEL, max_tokens=800):
    """Call OpenAI ChatCompletion API."""
    key = st.session_state.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OpenAI API key not set")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role":"system","content":"You are an expert clinical guideline summarizer. Answer only with JSON if requested."},
            {"role":"user","content":prompt}
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise Exception(f"OpenAI API call failed: {e}")

def run_llm_or_simulate(prompt, simulate_if_no_key=True):
    """Run LLM or simulate response if no API key available."""
    if st.session_state.get("openai_api_key") or os.environ.get("OPENAI_API_KEY"):
        try:
            resp = call_openai_chat(prompt)
            content = resp["choices"][0]["message"]["content"]
            # try to extract JSON object from response
            first = content.find("{")
            last = content.rfind("}")
            if first != -1 and last != -1:
                return {"ok": True, "json": json.loads(content[first:last+1]), "raw": content}
            else:
                return {"ok": False, "error": "No JSON in model output", "raw": content}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    else:
        if not simulate_if_no_key:
            return {"ok": False, "error": "No API key and simulate_if_no_key=False"}

        # deterministic simulation based on prompt content
        lower = prompt.lower()
        update = False
        rationale = "Simulation: insufficient new evidence found in retrieved abstracts."

        # Simple heuristic for simulation
        if any(word in lower for word in ["improve", "better", "effective", "superior", "significant"]):
            if "buprenorphine" in lower:
                update = True
                rationale = "Simulation: retrieved abstracts show improved retention with buprenorphine formulations."
            elif "naltrexone" in lower:
                update = True
                rationale = "Simulation: evidence supports naltrexone for patients completing detoxification."
            elif "methadone" in lower:
                update = True
                rationale = "Simulation: methadone shows sustained effectiveness in maintenance therapy."

        sim = {
            "update": update,
            "recommendation_text": "Consider new extended-release formulations for improved adherence and retention." if update else None,
            "strength": "Moderate" if update else None,
            "rationale": rationale,
            "evidence_used": ["SIM001", "SIM002"],
            "evidence_summary": "Simulated evidence summary based on keyword analysis of retrieved abstracts."
        }
        return {"ok": True, "json": sim, "raw": "SIMULATED"}

# -------------------------
# UTIL: compare texts (similarity + diff)
# -------------------------
def text_similarity(a, b):
    """Calculate similarity between two texts (0-1)."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def text_unified_diff(a, b, fromfile="baseline", tofile="candidate"):
    """Generate unified diff between two texts."""
    if not a:
        a = ""
    if not b:
        b = ""

    a_lines = a.splitlines(keepends=True)
    b_lines = b.splitlines(keepends=True)
    return "".join(unified_diff(a_lines, b_lines, fromfile=fromfile, tofile=tofile))

# -------------------------
# MAIN: run pipeline for a drug
# -------------------------
def run_pipeline_for_drug(drug_name: str, simulate_llm_if_no_key=True):
    """Main pipeline function for drug guideline analysis."""
    drug = drug_name.strip()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1) Download ASAM PDFs with fallback mechanism
    status_text.text("1. Downloading ASAM PDFs...")
    progress_bar.progress(10)
    
    try:
        asam2015_txt = download_pdf_text(ASAM_2015_PDF, FALLBACK_2015_PDF)
        st.success("✓ ASAM 2015 PDF downloaded successfully")
    except Exception as e:
        st.error(f"✗ Failed to download ASAM 2015 PDF: {e}")
        st.info("Using minimal placeholder text")
        asam2015_txt = f"ASAM 2015 Guideline - Placeholder text for {drug}. Original PDF unavailable due to: {e}"

    try:
        asam2020_txt = download_pdf_text(ASAM_2020_PDF, FALLBACK_2020_PDF)
        st.success("✓ ASAM 2020 PDF downloaded successfully")
    except Exception as e:
        st.error(f"✗ Failed to download ASAM 2020 PDF: {e}")
        st.info("Using minimal placeholder text")
        asam2020_txt = f"ASAM 2020 Guideline - Placeholder text for {drug}. Original PDF unavailable due to: {e}"

    progress_bar.progress(20)

    # 2) Extract relevant excerpts
    status_text.text("2. Extracting ASAM excerpts for drug...")
    asam2015_excerpts = extract_excerpts_by_keyword(asam2015_txt, drug, window=1200)
    asam2020_excerpts = extract_excerpts_by_keyword(asam2020_txt, drug, window=1200)

    st.info(f"Found {len(asam2015_excerpts)} excerpts in ASAM 2015")
    st.info(f"Found {len(asam2020_excerpts)} excerpts in ASAM 2020")

    # Create baseline texts
    baseline_2015 = "\n\n---\n\n".join(asam2015_excerpts) if asam2015_excerpts else f"(No ASAM 2015 excerpt found for {drug})"
    baseline_2020 = "\n\n---\n\n".join(asam2020_excerpts) if asam2020_excerpts else f"(No ASAM 2020 excerpt found for {drug})"

    progress_bar.progress(30)

    # 3) PubMed: search + fetch abstracts between 2015-2019
    status_text.text("3. Searching PubMed for abstracts (2015-2019)...")
    query = f'({drug}[MeSH] OR {drug}) AND ("Opioid Use Disorder"[MeSH] OR "Opioid Use Disorder")'
    pmids = pubmed_search_ids(query, retmax=NUM_PAPERS, mindate="2015", maxdate="2019")
    st.info(f"Found {len(pmids)} PMIDs")

    papers = pubmed_fetch_abstracts(pmids)
    st.info(f"Fetched {len(papers)} PubMed records with abstracts")

    progress_bar.progress(50)

    # 4) Build passages for RAG
    status_text.text("4. Building RAG passages...")
    passages = []
    meta = []

    # Add ASAM excerpts first
    if baseline_2015.strip() and not baseline_2015.startswith("(No ASAM"):
        passages.append(f"ASAM2015 | {drug} excerpt:\n\n{baseline_2015}")
        meta.append({"source":"ASAM2015","drug":drug})

    if baseline_2020.strip() and not baseline_2020.startswith("(No ASAM"):
        passages.append(f"ASAM2020 | {drug} excerpt:\n\n{baseline_2020}")
        meta.append({"source":"ASAM2020","drug":drug})

    # Add PubMed abstracts
    for p in papers:
        text = f"PMID:{p['pmid']} | {p['title']}\n\n{p['abstract']}"
        passages.append(text)
        meta.append({"source":"PubMed","pmid":p['pmid'],"year":p.get("year","")})

    st.info(f"Total passages: {len(passages)}")

    progress_bar.progress(60)

    # 5) Build embeddings + FAISS index
    status_text.text("5. Building embeddings and FAISS index...")
    if not passages:
        st.warning("No passages available for embedding")
        index, embs, model = None, None, None
    else:
        index, embs, model = build_embeddings_index(passages, model_name=EMBED_MODEL)
        st.success("Embeddings built successfully")

    progress_bar.progress(70)

    # 6) Prepare questions and run retrieval
    status_text.text("6. Running retrieval and analysis...")
    questions = [
        f"For adults with Opioid Use Disorder, does {drug} (including extended-release forms) improve treatment retention or relapse outcomes compared to alternatives between 2015-2019?",
        f"Based on the retrieved PubMed evidence (2015-2019) and ASAM 2015 baseline, should the ASAM guideline section on {drug} be updated?"
    ]

    results = []
    for i, q in enumerate(questions):
        st.write(f"**Question {i+1}:** {q}")

        if index is None:
            st.warning("Skipping - no index available")
            continue

        retrieved = retrieve_topk(index, model, q, passages, meta, k=TOP_K)
        st.info(f"Retrieved {len(retrieved)} passages")

        # Build prompt for LLM - FIXED: Removed backslashes from f-string
        evidence_texts = []
        evidence_ids = []
        for r in retrieved:
            m = r["meta"]
            if m.get("source") == "PubMed":
                evidence_texts.append(r["passage"])
                evidence_ids.append(f"PMID:{m.get('pmid')}")
            else:
                evidence_texts.append(r["passage"])
                evidence_ids.append(m.get("source"))

        baseline_snippet = baseline_2015 if baseline_2015 and not baseline_2015.startswith("(No ASAM") else "ASAM 2015 baseline not found for this drug."

        # Create prompt without backslashes in f-strings
        evidence_separator = "\n\n---\n\n"
        prompt = f"""
You are an expert clinician and evidence summarizer.

QUESTION:
{q}

CURRENT ASAM 2015 BASELINE (excerpt):
{baseline_snippet[:1800]}

RETRIEVED EVIDENCE (top passages):
{evidence_separator.join(evidence_texts[:3])}

TASK:
1) Based ONLY on the retrieved evidence and ASAM 2015 excerpt, indicate whether the guideline should be UPDATED (true/false).
2) If update==true, provide a concise recommendation (1-2 sentences) suitable for insertion into the guideline and indicate evidence strength (Strong/Moderate/Weak).
3) If update==false, set recommendation_text and strength to null and provide a short rationale.
4) Provide evidence_used as a list of PMIDs or ASAM tags.
5) Provide a 1-2 sentence evidence_summary.

Return ONLY a JSON object with keys:
update, recommendation_text, strength, rationale, evidence_used, evidence_summary
"""
        # Call LLM or simulate
        llm_out = run_llm_or_simulate(prompt, simulate_if_no_key=simulate_llm_if_no_key)
        if not llm_out.get("ok"):
            st.error(f"LLM failed: {llm_out.get('error')}")
            results.append({"question": q, "retrieved": retrieved, "llm": None, "error": llm_out.get("error")})
            continue

        parsed = llm_out["json"]
        # if evidence_used empty fill from retrieved evidence ids
        if not parsed.get("evidence_used"):
            parsed["evidence_used"] = evidence_ids[:3]  # Limit to top 3

        # attach provenance passages lightly
        parsed["_provenance"] = [{"meta": r["meta"], "snippet": r["passage"][:400]} for r in retrieved[:2]]
        results.append({"question": q, "retrieved": retrieved, "llm": parsed})

        st.success(f"LLM result: update={parsed.get('update')}")

    progress_bar.progress(85)

    # 7) Compare AI candidate update with ASAM 2020 excerpt
    status_text.text("7. Comparing with ASAM 2020...")
    candidate = None
    # prefer second question's result for update decision if present
    for r in results:
        if "should the ASAM guideline" in r["question"].lower():
            candidate = r.get("llm")
            break
    if candidate is None and results:
        candidate = results[0].get("llm")

    comp = {}
    if candidate:
        candidate_text = candidate.get("recommendation_text") or "(No candidate recommendation provided.)"
        asam2020_text = baseline_2020 if baseline_2020 and not baseline_2020.startswith("(No ASAM") else ""

        sim_score = text_similarity(candidate_text, asam2020_text) if asam2020_text else None
        diff_text = text_unified_diff(asam2020_text, candidate_text, fromfile="ASAM2020", tofile="AI_candidate")

        comp = {
            "candidate_text": candidate_text,
            "asam2020_text": asam2020_text[:4000],
            "similarity": sim_score,
            "diff": diff_text
        }

        st.info(f"Similarity score: {sim_score:.3f}")
    else:
        st.warning("No candidate produced by LLM to compare.")

    progress_bar.progress(100)
    status_text.text("Analysis complete!")

    # 8) Final packaging and return
    out = {
        "drug": drug,
        "timestamp": datetime.utcnow().isoformat(),
        "pmids": [p["pmid"] for p in papers],
        "papers": papers,
        "asam2015_excerpt": baseline_2015,
        "asam2020_excerpt": baseline_2020,
        "rag_llm_results": results,
        "comparison": comp
    }

    return out

# -------------------------
# STREAMLIT UI
# -------------------------
def main():
    st.title("📚 ASAM Living Guideline Analyzer")
    st.markdown("""
    This tool analyzes drugs for Opioid Use Disorder treatment guidelines by:
    - Searching PubMed for recent evidence (2015-2019)
    - Comparing with ASAM 2015 baseline guidelines
    - Generating evidence-based recommendations
    - Comparing with actual ASAM 2020 updates
    """)

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password", 
                                   help="Leave empty to use simulation mode")
    if api_key:
        st.session_state.openai_api_key = api_key
        st.sidebar.success("✅ API Key set")
    else:
        st.sidebar.info("🔒 Using simulation mode (no API key)")

    # Drug selection
    st.sidebar.header("Drug Selection")
    drug_options = ["Buprenorphine", "Methadone", "Naltrexone", "Other"]
    selected_drug = st.sidebar.selectbox("Select a drug:", drug_options)
    
    if selected_drug == "Other":
        custom_drug = st.sidebar.text_input("Enter custom drug name:")
        drug_name = custom_drug.strip() if custom_drug else None
    else:
        drug_name = selected_drug

    # Analysis button
    if st.sidebar.button("🚀 Run Analysis", disabled=not drug_name):
        if drug_name:
            with st.spinner("Running analysis... This may take a few minutes."):
                result = run_pipeline_for_drug(drug_name, simulate_llm_if_no_key=not api_key)
            
            # Display results
            st.header("📊 Results Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Drug Analyzed", result['drug'])
            with col2:
                st.metric("PubMed Abstracts", len(result['papers']))
            with col3:
                if result["comparison"]:
                    sim_score = result["comparison"].get("similarity", 0)
                    st.metric("Similarity Score", f"{sim_score:.3f}")

            # Display LLM results
            st.header("🤖 AI Recommendations")
            if result["rag_llm_results"]:
                for idx, r in enumerate(result["rag_llm_results"], start=1):
                    with st.expander(f"Question {idx}: {r['question'][:100]}..."):
                        if r.get("llm"):
                            llm_data = r['llm']
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Update Recommended:**", "✅ Yes" if llm_data.get('update') else "❌ No")
                                if llm_data.get('strength'):
                                    st.write("**Evidence Strength:**", llm_data.get('strength'))
                                st.write("**Evidence Used:**", ", ".join(llm_data.get('evidence_used', [])))
                            
                            with col2:
                                if llm_data.get('recommendation_text'):
                                    st.write("**Recommendation:**")
                                    st.info(llm_data.get('recommendation_text'))
                            
                            st.write("**Rationale:**", llm_data.get('rationale'))
                            st.write("**Evidence Summary:**", llm_data.get('evidence_summary'))

            # Display comparison
            if result["comparison"]:
                st.header("🔍 Comparison with ASAM 2020")
                comp = result["comparison"]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("AI Recommendation")
                    st.write(comp.get('candidate_text', 'No recommendation'))
                
                with col2:
                    st.subheader("Actual ASAM 2020")
                    st.write(comp.get('asam2020_text', 'No ASAM 2020 text found'))
                
                # Similarity interpretation
                sim_score = comp.get('similarity', 0)
                st.subheader("Similarity Analysis")
                
                if sim_score > 0.7:
                    st.success(f"High similarity ({sim_score:.3f}) - AI recommendation aligns well with actual guidelines")
                elif sim_score > 0.4:
                    st.info(f"Moderate similarity ({sim_score:.3f}) - Some alignment with actual guidelines")
                elif sim_score > 0.2:
                    st.warning(f"Low similarity ({sim_score:.3f}) - Limited alignment with actual guidelines")
                else:
                    st.error(f"Very low similarity ({sim_score:.3f}) - Poor alignment with actual guidelines")

            # Download results
            st.header("💾 Download Results")
            json_str = json.dumps(result, indent=2)
            st.download_button(
                label="Download JSON Results",
                data=json_str,
                file_name=f"guideline_analysis_{drug_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        else:
            st.error("Please enter a drug name to analyze.")

    # Instructions
    st.sidebar.header("Instructions")
    st.sidebar.markdown("""
    1. Select or enter a drug name
    2. Optional: Add OpenAI API key for real LLM analysis
    3. Click 'Run Analysis'
    4. View results and download data
    """)

if __name__ == "__main__":
    main()