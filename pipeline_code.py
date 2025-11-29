# pipeline_code.py
# Hybrid pipeline (OA PDFs + PubMed abstracts fallback)
# Produces AI answers + AI summary. Reviewer route (external) will consume review_token and finalize.

import io
import json
import time
import uuid
import re
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

# optional pdf libs
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import fitz
except Exception:
    fitz = None

# optional embedding libs
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except Exception:
    SentenceTransformer = None
    faiss = None

# -----------------------
# CONFIG
# -----------------------
OUTDIR = Path("living_guideline_run")
OUTDIR.mkdir(exist_ok=True)
FULLTEXT_DIR = OUTDIR / "fulltexts"
FULLTEXT_DIR.mkdir(exist_ok=True)
REVIEW_FILE = OUTDIR / "reviews.json"

NUM_PAPERS = 200
TOP_K = 5
EMBED_MODEL = "all-MiniLM-L6-v2"

OPENAI_API_KEY = None  # set via UI set_openai_key()

# PubMed fixed query (Phase I-IV 2015-2019)
PUBMED_QUERY = (
    '"opioid-related disorders"[MeSH Terms] AND '
    '("2015/01/01"[PDAT] : "2019/12/31"[PDAT]) AND '
    '('
    '"Clinical Trial"[PT] OR '
    '"Clinical Trial, Phase I"[PT] OR '
    '"Clinical Trial, Phase II"[PT] OR '
    '"Clinical Trial, Phase III"[PT] OR '
    '"Clinical Trial, Phase IV"[PT]'
    ')'
)
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# -----------------------
# UTILS
# -----------------------
def now():
    return datetime.utcnow().isoformat()

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def load_json(path):
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

def set_openai_key(key):
    global OPENAI_API_KEY
    OPENAI_API_KEY = key.strip() if key else None

# -----------------------
# PDF extraction
# -----------------------
def extract_pdf_text(data_bytes: bytes) -> str:
    text = ""
    if pdfplumber:
        try:
            with pdfplumber.open(io.BytesIO(data_bytes)) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
                text = "\n\n".join(pages).strip()
                if len(text) > 50:
                    return text
        except Exception:
            pass
    if fitz:
        try:
            doc = fitz.open(stream=data_bytes, filetype="pdf")
            pages = [p.get_text() for p in doc]
            text = "\n\n".join(pages).strip()
            if len(text) > 50:
                return text
        except Exception:
            pass
    return ""

# -----------------------
# PubMed / EuropePMC
# -----------------------
def pubmed_search(retmax=NUM_PAPERS):
    params = {"db": "pubmed", "term": PUBMED_QUERY, "retmax": retmax, "retmode": "json"}
    try:
        r = requests.get(f"{PUBMED_BASE}/esearch.fcgi", params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []

def fetch_pubmed_records(pmids):
    if not pmids:
        return []
    ids = ",".join(pmids)
    params = {"db": "pubmed", "id": ids, "retmode": "xml"}
    try:
        r = requests.get(f"{PUBMED_BASE}/efetch.fcgi", params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception:
        return []
    recs = []
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID") or ""
        title = art.findtext(".//ArticleTitle") or ""
        abs_text = " ".join(ET.tostring(n, encoding="unicode", method="text").strip() for n in art.findall(".//AbstractText")).strip()
        journal = art.findtext(".//Journal/Title") or ""
        year = art.findtext(".//PubDate/Year") or ""
        doi = None
        for aid in art.findall(".//ArticleId"):
            if aid.get("IdType","").lower() == "doi":
                doi = aid.text
        recs.append({"pmid": pmid, "title": title, "abstract": abs_text, "journal": journal, "year": year, "doi": doi})
    return recs

def fetch_europepmc_fulltext_links(pmid=None, doi=None):
    if not pmid and not doi:
        return []
    q = f"EXT_ID:{pmid}" if pmid else f"DOI:{doi}"
    params = {"query": q, "format": "json", "resultType": "core", "pageSize": 1}
    try:
        r = requests.get(EUROPEPMC_SEARCH, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        results = data.get("resultList", {}).get("result", [])
        if not results:
            return []
        item = results[0]
        urls = [f.get("url") for f in item.get("fullTextUrlList", {}).get("fullTextUrl", []) if f.get("url")]
        return urls
    except Exception:
        return []

def attempt_fulltext_download(paper):
    pmid = paper.get("pmid")
    doi = paper.get("doi")
    links = fetch_europepmc_fulltext_links(pmid=pmid, doi=doi)
    if not links:
        return ""
    for url in links:
        try:
            r = requests.get(url, timeout=40)
            ct = (r.headers.get("content-type") or "").lower()
            if "pdf" in ct or url.lower().endswith(".pdf"):
                text = extract_pdf_text(r.content)
                if text and len(text) > 200:
                    try:
                        Path(FULLTEXT_DIR / f"{pmid}.pdf").write_bytes(r.content)
                    except Exception:
                        pass
                    return text
            html = r.text or ""
            plain = re.sub(r"<[^>]+>", " ", html)
            plain = re.sub(r"\s+", " ", plain).strip()
            if len(plain) > 300:
                return plain
        except Exception:
            continue
    return ""

# -----------------------
# Small LLM wrapper (OpenAI chat or deterministic simulation)
# -----------------------
def call_openai_chat(prompt, model="gpt-4o-mini", max_tokens=800):
    if not OPENAI_API_KEY:
        # deterministic/safe simulation
        lower = (prompt or "").lower()
        update = any(w in lower for w in ["improve", "effective", "better", "superior", "significant"])
        return {
            "update": bool(update),
            "recommendation_text": "Simulated: consider extended-release formulation" if update else None,
            "strength": "Moderate" if update else None,
            "rationale": "Simulated rationale (no API key).",
            "evidence_used": [],
            "evidence_summary": "Simulated evidence summary."
        }
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages":[{"role":"system","content":"You are an expert clinical guideline summarizer. Return only JSON when asked."},{"role":"user","content":prompt}], "temperature": 0.0, "max_tokens": max_tokens}
    for attempt in range(3):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=90)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            first = content.find("{"); last = content.rfind("}")
            if first != -1 and last != -1:
                try:
                    return json.loads(content[first:last+1])
                except Exception:
                    return {"update": None, "rationale": "LLM returned non-JSON"}
            return {"update": None, "rationale": content}
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 429:
                time.sleep(2**attempt)
                continue
            return {"update": None, "rationale": f"OpenAI HTTPError: {e}"}
        except Exception as e:
            return {"update": None, "rationale": f"OpenAI error: {e}"}
    return {"update": None, "rationale": "OpenAI retries exhausted."}

# -----------------------
# Generate AI-written 1-2 paragraph summary from per-question answers
# -----------------------
def generate_ai_summary(results, drug):
    # build a concise context for the LLM
    brief = []
    for r in results:
        q = r.get("_question","")
        rec = r.get("recommendation_text") or ""
        strength = r.get("strength") or ""
        brief.append(f"Q: {q}\nRec: {rec}\nStrength: {strength}")
    prompt = f"""
You are an expert clinical summarizer. Based on the following per-question recommendations for drug '{drug}', write a 1-2 paragraph clinical summary (3-6 sentences each paragraph) that could be used as a guideline update summary. Be concise and cite which questions informed the summary (use plain PMID lists when available).

ITEMS:
{chr(10).join(brief)}

Return ONLY JSON: {{ "ai_summary": "<two short paragraphs>" }}
"""
    out = call_openai_chat(prompt)
    if isinstance(out, dict) and out.get("ai_summary"):
        return out.get("ai_summary")
    # fallback: simple composition
    positives = [r for r in results if r.get("update") is True]
    if positives:
        return "AI Summary (simulated): Some questions indicated updates recommending the intervention; consider reviewing extended-release formulations and dosing. Reviewer must finalize."
    return "AI Summary (simulated): No major updates suggested by the retrieved trials. Reviewer must finalize."

# -----------------------
# REVIEW storage helpers
# -----------------------
def add_review_token(drug, results, conflicts):
    token = str(uuid.uuid4())
    reviews = load_json(REVIEW_FILE)
    reviews[token] = {
        "drug": drug,
        "results": results,
        "conflicts": conflicts,
        "status": "pending",   # pending / approved / rejected / needs_changes
        "ai_summary": generate_ai_summary(results, drug),
        "reviewer_summary": None,
        "reviewer_email": None,
        "reviewer_decision": None,
        "reviewer_notes": None,
        "created_at": now(),
    }
    save_json(REVIEW_FILE, reviews)
    return token

# -----------------------
# Main pipeline (hybrid) — returns complete structure always
# -----------------------
def run_pipeline_for_drug(drug_name, questions, progress_callback=None):
    """
    progress_callback(step_idx:int, message:str, progress:float)
    Returns full output dict (never None)
    """
    if progress_callback is None:
        def _nop(a,b,c): pass
        progress_callback = _nop

    drug = (drug_name or "").strip()
    ts = now()
    output = {
        "drug": drug,
        "timestamp": ts,
        "pmids": [],
        "papers": [],
        "results": [],
        "classifications": [],
        "conflicts": {"has_conflict": False, "positive": [], "negative": [], "neutral": []},
        "summary": {"drug": drug, "total_questions": len(questions or []), "update_yes": 0, "update_no": len(questions or []), "final_recommendation": "No Evidence Found"},
        "review_token": None
    }

    try:
        progress_callback(0, "Searching PubMed...", 0.02)
        pmids = pubmed_search()
        output["pmids"] = pmids

        progress_callback(5, f"Found {len(pmids)} PMIDs; fetching records...", 0.08)
        records = fetch_pubmed_records(pmids)
        output["papers"] = records

        progress_callback(10, "Attempting OA fulltext downloads (Europe PMC)...", 0.12)
        for i,p in enumerate(records):
            progress_callback(10 + int(i/ max(1,len(records)) * 10), f"Downloading PMID {p.get('pmid')}", 0.12 + (i/ max(1,len(records)) * 0.12))
            try:
                ft = attempt_fulltext_download(p)
                if ft:
                    p["fulltext_text"] = ft
                    p["used_fulltext"] = True
                else:
                    p["fulltext_text"] = ""
                    p["used_fulltext"] = False
            except Exception:
                p["fulltext_text"] = ""
                p["used_fulltext"] = False

        passages = []
        meta = []
        for p in records:
            txt = (p.get("fulltext_text") or "").strip() or (p.get("abstract") or "").strip()
            if txt:
                passages.append(txt)
                meta.append({"pmid": p.get("pmid"), "title": p.get("title")})

        if not passages:
            output["summary"]["final_recommendation"] = f"No text available ({len(records)} records)."
            save_json(OUTDIR / f"pipeline_{drug or 'unknown'}.json", output)
            progress_callback(1.0, "No passages available — pipeline ended", 1.0)
            return output

        # simple retrieval fallback (no embeddings required)
        def retrieve_simple(question, k=TOP_K):
            qwords = set(re.findall(r"\w+", question.lower()))
            scored = []
            for i, txt in enumerate(passages):
                score = sum(1 for w in qwords if w in txt.lower())
                scored.append((score, i))
            scored.sort(reverse=True)
            res = []
            for s, idx in scored[:k]:
                res.append({"score": float(s), "passage": passages[idx], "meta": meta[idx]})
            return res

        progress_callback(50, "Running question-answering using LLM...", 0.55)
        results = []
        for qi, q in enumerate(questions or []):
            progress_callback(50 + int((qi/ max(1, len(questions or []))) * 30), f"Retrieving & answering Q{qi+1}", 0.55 + (qi/ max(1, len(questions or [])) * 0.30))
            retrieved = retrieve_simple(q, k=TOP_K)
            evidence_texts = [r["passage"] for r in retrieved[:TOP_K]]
            evidence_ids = [r["meta"].get("pmid") for r in retrieved[:TOP_K]]
            joined = "\n\n---\n\n".join(evidence_texts[:5])

            prompt = f"""
You are an expert clinical guideline summarizer.

QUESTION:
{q}

EVIDENCE (top passages):
{joined}

INSTRUCTIONS:
Answer ONLY with a JSON object:
{{ "update": true/false/null, "recommendation_text": "one-sentence recommendation or null", "strength": "Strong|Moderate|Weak|null", "rationale": "one-sentence rationale", "evidence_used": ["PMID1","PMID2"], "evidence_summary": "short summary" }}
"""
            ans = call_openai_chat(prompt)
            if not isinstance(ans, dict):
                ans = {"update": None, "recommendation_text": None, "strength": None, "rationale": str(ans), "evidence_used": evidence_ids, "evidence_summary": ""}
            if not ans.get("evidence_used"):
                ans["evidence_used"] = [e for e in evidence_ids if e]
            ans["_question"] = q
            ans["_retrieved"] = retrieved
            results.append(ans)

        output["results"] = results

        # classify stances (per paper) — simple simulation via LLM
        progress_callback(85, "Classifying paper stances...", 0.88)
        classifications = []
        for i,p in enumerate(records):
            txt = (p.get("fulltext_text") or p.get("abstract") or "")[:3000]
            if not txt:
                classifications.append({"pmid": p.get("pmid"), "title": p.get("title"), "stance": "neutral", "summary": ""})
                continue
            try:
                stn, summ = classify_paper_stance(txt, drug)
            except Exception:
                stn, summ = "neutral", ""
            classifications.append({"pmid": p.get("pmid"), "title": p.get("title"), "stance": stn, "summary": summ})

        output["classifications"] = classifications
        conflicts = detect_conflicts(classifications)
        output["conflicts"] = conflicts

        # add review token if conflicts or if any updates suggested
        upd_yes = sum(1 for r in results if r.get("update") is True)
        if conflicts.get("has_conflict") or upd_yes > 0:
            token = add_review_token(drug, results, conflicts)
            output["review_token"] = token

        # AI summary generation
        ai_summary = generate_ai_summary(results, drug)
        output["summary"] = {"drug": drug, "total_questions": len(questions or []), "update_yes": int(upd_yes), "update_no": int(len(questions or []) - upd_yes), "ai_summary": ai_summary, "final_recommendation": "Pending reviewer"}
        save_json(OUTDIR / f"pipeline_{drug or 'unknown'}.json", output)
        progress_callback(1.0, "Pipeline complete", 1.0)
        return output

    except Exception as e:
        # safe error return
        output["error"] = str(e)
        save_json(OUTDIR / f"pipeline_error_{drug or 'unknown'}.json", output)
        progress_callback(1.0, f"Pipeline error: {e}", 1.0)
        return output

# -----------------------
# small helpers used above (stance classification)
# -----------------------
def classify_paper_stance(paper_text, drug_name):
    prompt = f"""
Classify the stance of this paper text regarding the drug '{drug_name}' for Opioid Use Disorder.
Return ONLY JSON: {{ "stance": "positive|negative|neutral", "summary": "one-sentence" }}

Text:
{paper_text[:3500]}
"""
    out = call_openai_chat(prompt)
    if isinstance(out, dict):
        return (out.get("stance") or "neutral", out.get("summary") or "")
    return ("neutral", "")

def detect_conflicts(classifications):
    pos = [c for c in classifications if c.get("stance") == "positive"]
    neg = [c for c in classifications if c.get("stance") == "negative"]
    return {"has_conflict": (len(pos)>0 and len(neg)>0), "positive": pos, "negative": neg, "neutral": [c for c in classifications if c.get("stance")=="neutral"]}

# -----------------------
# helper for sending emails removed from pipeline; streamlit app sends emails
# -----------------------
# End of pipeline_code.py
