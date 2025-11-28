# pipeline_code.py
# Hybrid pipeline (OA PDFs + PubMed abstracts fallback) with progress callback
# Always returns a complete structure. Call run_pipeline_for_drug(..., progress_callback=cb)

import os
import io
import json
import time
import uuid
import re
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

# Optional PDF libs; if absent, pipeline still works (falls back to abstracts)
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# optional embedding libs
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except Exception:
    SentenceTransformer = None
    faiss = None

# ---------------------
# CONFIG
# ---------------------
OUTDIR = Path("living_guideline_run")
OUTDIR.mkdir(exist_ok=True)
FULLTEXT_DIR = OUTDIR / "fulltexts"
FULLTEXT_DIR.mkdir(exist_ok=True)
REVIEW_FILE = OUTDIR / "reviews.json"

NUM_PAPERS = 200
TOP_K = 5
EMBED_MODEL = "all-MiniLM-L6-v2"

OPENAI_API_KEY = None  # set by UI via set_openai_key()

# PubMed query (user-specified / accepted)
PUBMED_QUERY =  (
        '"opioid-related disorders"[MeSH Terms] AND '
        '("2015/01/01"[Publication Date] : "2019/12/31"[Publication Date]) AND '
        '("Clinical Trial, Phase I"[Article Type] OR '
        '"Clinical Trial, Phase II"[Article Type] OR '
        '"Clinical Trial, Phase III"[Article Type] OR '
        '"Clinical Trial, Phase IV"[Article Type])'
    )

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# ---------------------
# UTILITIES
# ---------------------
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

# ---------------------
# PDF text extraction
# ---------------------
def extract_pdf_text(data_bytes: bytes) -> str:
    """Try pdfplumber then fitz. Return text or empty string."""
    text = ""
    if pdfplumber:
        try:
            with pdfplumber.open(io.BytesIO(data_bytes)) as pdf:
                pages = []
                for p in pdf.pages:
                    pages.append(p.extract_text() or "")
                text = "\n\n".join(pages).strip()
                if len(text) > 50:
                    return text
        except Exception:
            pass

    if fitz:
        try:
            doc = fitz.open(stream=data_bytes, filetype="pdf")
            pages = [page.get_text() for page in doc]
            text = "\n\n".join(pages).strip()
            if len(text) > 50:
                return text
        except Exception:
            pass

    return ""

# ---------------------
# PubMed / Europe PMC
# ---------------------
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

    records = []
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID") or ""
        title = art.findtext(".//ArticleTitle") or ""
        abs_nodes = art.findall(".//AbstractText")
        abstract = " ".join(ET.tostring(n, encoding="unicode", method="text").strip() for n in abs_nodes).strip()
        journal = art.findtext(".//Journal/Title") or ""
        year = art.findtext(".//PubDate/Year") or ""
        doi = None
        for aid in art.findall(".//ArticleId"):
            if aid.get("IdType", "").lower() == "doi":
                doi = aid.text
        records.append({"pmid": pmid, "title": title, "abstract": abstract, "journal": journal, "year": year, "doi": doi})
    return records

def fetch_europepmc_fulltext_links(pmid=None, doi=None):
    if not pmid and not doi:
        return []
    q = f"EXT_ID:{pmid}" if pmid else f"DOI:{doi}"
    params = {"query": q, "format": "json", "resultType": "core", "pageSize": 1}
    try:
        r = requests.get(EUROPEPMC_SEARCH, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        res = data.get("resultList", {}).get("result", [])
        if not res:
            return []
        item = res[0]
        urls = []
        for f in item.get("fullTextUrlList", {}).get("fullTextUrl", []):
            url = f.get("url")
            if url:
                urls.append(url)
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
            # html fallback
            html = r.text or ""
            plain = re.sub(r"<[^>]+>", " ", html)
            plain = re.sub(r"\s+", " ", plain).strip()
            if len(plain) > 300:
                return plain
        except Exception:
            continue
    return ""

# ---------------------
# embeddings + retrieval (if available)
# ---------------------
def build_embeddings_index(passages):
    if SentenceTransformer is None or faiss is None:
        raise RuntimeError("sentence-transformers or faiss not installed")
    model = SentenceTransformer(EMBED_MODEL)
    embs = model.encode(passages, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatL2(embs.shape[1])
    index.add(embs)
    return index, model

def retrieve_topk(index, model, question, passages, meta, k=TOP_K):
    q_emb = model.encode([question], convert_to_numpy=True).astype("float32")
    k = min(k, len(passages))
    D, I = index.search(q_emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < len(passages):
            results.append({"score": float(score), "passage": passages[idx], "meta": meta[idx]})
    return results

# ---------------------
# LLM wrapper with simple retry
# ---------------------
def call_openai_chat(prompt, model="gpt-4o-mini", max_tokens=800):
    if not OPENAI_API_KEY:
        # deterministic simulation fallback
        lw = (prompt or "").lower()
        upd = any(w in lw for w in ["improve", "effective", "better", "superior", "significant"])
        return {"update": bool(upd), "recommendation_text": "Simulated: consider update" if upd else None, "strength": "Moderate" if upd else None, "rationale": "Simulated (no API key)", "evidence_used": [], "evidence_summary": "Simulated summary."}

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "system", "content": "You are an expert clinical guideline summarizer. Return only JSON when requested."}, {"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": max_tokens}

    for attempt in range(3):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=90)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            first = content.find("{")
            last = content.rfind("}")
            if first != -1 and last != -1:
                try:
                    return json.loads(content[first:last+1])
                except Exception:
                    return {"update": None, "rationale": "LLM returned non-JSON"}
            return {"update": None, "rationale": content}
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 429:
                time.sleep(2 ** attempt)
                continue
            return {"update": None, "rationale": f"OpenAI HTTPError: {e}"}
        except Exception as e:
            return {"update": None, "rationale": f"OpenAI error: {e}"}
    return {"update": None, "rationale": "OpenAI retries exhausted"}

# ---------------------
# paper stance classification
# ---------------------
def classify_paper_stance(paper_text, drug_name):
    prompt = f"""Classify the stance of this paper regarding the drug '{drug_name}' for Opioid Use Disorder.
Return ONLY JSON: {{ "stance": "positive|negative|neutral", "summary": "one-sentence summary" }}
Text:
{paper_text[:3500]}
"""
    out = call_openai_chat(prompt)
    stance = out.get("stance") if isinstance(out, dict) else None
    summary = out.get("summary") if isinstance(out, dict) else None
    return (stance or "neutral", summary or "")

def detect_conflicts(classifications):
    pos = [c for c in classifications if c.get("stance") == "positive"]
    neg = [c for c in classifications if c.get("stance") == "negative"]
    return {"has_conflict": (len(pos) > 0 and len(neg) > 0), "positive": pos, "negative": neg, "neutral": [c for c in classifications if c.get("stance") == "neutral"]}

# ---------------------
# MAIN pipeline with progress_callback
# ---------------------
def run_pipeline_for_drug(drug_name, questions, progress_callback=None, simulate_if_no_key=True):
    """
    progress_callback: function(step_idx:int, message:str, progress:float) - progress in [0,1]
    """
    if progress_callback is None:
        def _nop(a,b,c): pass
        progress_callback = _nop

    ts = now()
    drug = (drug_name or "").strip()
    drug_lower = drug.lower()

    # prepare safe output skeleton
    output = {
        "drug": drug,
        "timestamp": ts,
        "pmids": [],
        "papers": [],
        "results": [],
        "classifications": [],
        "conflicts": {"has_conflict": False, "positive": [], "negative": [], "neutral": []},
        "summary": {"drug": drug, "total_questions": len(questions or []), "update_yes": 0, "update_no": len(questions or []), "final_recommendation": "No Evidence Found"},
    }

    try:
        progress_callback(0, "Searching PubMed (Phase I–IV trials 2015–2019)...", 0.02)
        pmids = pubmed_search()
        output["pmids"] = pmids

        progress_callback(1, f"Found {len(pmids)} PMIDs. Fetching records...", 0.08)
        records = fetch_pubmed_records(pmids)
        output["papers"] = records

        progress_callback(2, "Attempting to download OA full texts (if available)...", 0.12)
        for i, p in enumerate(records):
            # small progress per record
            progress_callback(10 + int((i/ max(1,len(records))) * 10), f"Downloading OA fulltext for PMID {p.get('pmid')}...", 0.12 + (i/ max(1,len(records)) * 0.18))
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

        progress_callback(25, "Building passage list (prefer fulltext, fallback to abstract)...", 0.35)
        passages = []
        meta = []
        for p in records:
            txt = (p.get("fulltext_text") or "").strip() or (p.get("abstract") or "").strip()
            if txt:
                passages.append(txt)
                meta.append({"pmid": p.get("pmid"), "title": p.get("title")})

        if not passages:
            output["summary"]["final_recommendation"] = f"No text passages available ({len(records)} records)."
            save_json(OUTDIR / f"pipeline_{drug or 'unknown'}.json", output)
            progress_callback(1.0, "No passages available — pipeline ended.", 1.0)
            return output

        # attempt embedding index build
        use_embeddings = (SentenceTransformer is not None and faiss is not None)
        if use_embeddings:
            try:
                progress_callback(40, "Building embeddings index...", 0.45)
                index, emb_model = build_embeddings_index(passages)
                progress_callback(45, "Embeddings built.", 0.5)
            except Exception as e:
                print("Embeddings error:", e)
                use_embeddings = False
                index = None
                emb_model = None
        else:
            index = None
            emb_model = None

        # retrieval function
        def _retrieve(q, k=TOP_K):
            if use_embeddings and index and emb_model:
                return retrieve_topk(index, emb_model, q, passages, meta, k=k)
            # naive fallback: count overlapping words
            qwords = set(re.findall(r"\w+", q.lower()))
            scored = []
            for i, txt in enumerate(passages):
                score = sum(1 for w in qwords if w in txt.lower())
                scored.append((score, i))
            scored.sort(reverse=True)
            out = []
            for s, idx in scored[:k]:
                out.append({"score": float(s), "passage": passages[idx], "meta": meta[idx]})
            return out

        # run Q&A for each question
        results = []
        total_q = len(questions or [])
        for qi, q in enumerate(questions or []):
            progress_callback(50 + int((qi/ max(1,total_q))*30), f"Retrieving evidence for question {qi+1}/{total_q}...", 0.55 + (qi/ max(1,total_q) * 0.30))
            retrieved = _retrieve(q, k=TOP_K)
            evidence_texts = [r["passage"] for r in retrieved[:TOP_K]]
            evidence_ids = [r["meta"].get("pmid") for r in retrieved[:TOP_K]]
            joined = "\n\n---\n\n".join(evidence_texts[:5])

            prompt = f"""You are an expert clinical guideline summarizer.
QUESTION: {q}
EVIDENCE:
{joined}
Return ONLY a JSON object with keys:
update (true/false), recommendation_text (string|null), strength (Strong|Moderate|Weak|null), rationale (string), evidence_used (list of PMIDs), evidence_summary (string)
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
        progress_callback(85, "Classifying paper stances...", 0.88)

        classifications = []
        for i, p in enumerate(records):
            progress_callback(86 + int((i / max(1,len(records))) * 10), f"Classifying stance for PMID {p.get('pmid')}", 0.88 + (i / max(1,len(records)) * 0.08))
            txt = (p.get("fulltext_text") or p.get("abstract") or "").strip()
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

        upd_yes = sum(1 for r in results if r.get("update") is True)
        output["summary"] = {"drug": drug, "total_questions": len(questions or []), "update_yes": int(upd_yes), "update_no": int(len(questions or []) - upd_yes), "final_recommendation": "Update Recommended" if upd_yes > (len(questions or []) / 2) else "No Major Update"}

        # create review token if conflict
        if conflicts.get("has_conflict"):
            token = str(uuid.uuid4())
            output["review_token"] = token
            reviews = load_json(REVIEW_FILE)
            reviews[token] = {"drug": drug, "results": results, "conflicts": conflicts, "status": "pending", "created_at": ts}
            save_json(REVIEW_FILE, reviews)

        save_json(OUTDIR / f"pipeline_{drug or 'unknown'}.json", output)
        progress_callback(1.0, "Pipeline complete.", 1.0)
        return output

    except Exception as e:
        # on any exception, return safe error object
        err_out = {
            "drug": drug,
            "timestamp": now(),
            "pmids": output.get("pmids", []),
            "papers": output.get("papers", []),
            "results": output.get("results", []),
            "classifications": output.get("classifications", []),
            "conflicts": output.get("conflicts", {}),
            "summary": output.get("summary", {"drug":drug, "total_questions": len(questions or []), "update_yes":0, "update_no": len(questions or []), "final_recommendation": "Error"}),
            "error": str(e)
        }
        save_json(OUTDIR / f"pipeline_error_{drug or 'unknown'}.json", err_out)
        progress_callback(1.0, f"Pipeline error: {e}", 1.0)
        return err_out
