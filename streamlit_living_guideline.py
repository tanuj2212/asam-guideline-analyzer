# streamlit_living_guideline.py
import streamlit as st
import json
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from pipeline_code import run_pipeline_for_drug, set_openai_key, load_json, REVIEW_FILE, OUTDIR

st.set_page_config(page_title="ASAM Living Guideline Updater (Hybrid + Reviewer)", layout="wide")
st.title("ASAM Living Guideline Updater — Hybrid (OA PDFs + Abstracts)")
st.caption("2015–2019 Phase I–IV clinical trials for opioid-related disorders. Reviewer route + email notifications included.")

# -----------------------------
# Sidebar: OpenAI key, SMTP settings, questions
# -----------------------------
with st.sidebar:
    st.header("Configuration")

    api_key = st.text_input("OpenAI API Key (optional)", type="password")
    if st.button("Set OpenAI Key"):
        set_openai_key(api_key)
        st.success("OpenAI key set (or simulation mode if blank).")

    st.markdown("---")
    st.subheader("SMTP (for review emails)")
    smtp_server = st.text_input("SMTP server", value="smtp.gmail.com")
    smtp_port = st.number_input("SMTP port", value=465)
    smtp_username = st.text_input("SMTP username (email)")
    smtp_password = st.text_input("SMTP password or app password", type="password")
    sender_name = st.text_input("Sender name", value="Guideline Bot")
    

    st.markdown("---")
    st.subheader("Questions")
    use_default = st.checkbox("Use default 10 questions", value=True)
    default_questions = [
        "For adults with OUD, does the drug improve treatment retention vs alternatives?",
        "Does evidence show reduced relapse with the drug?",
        "Are extended-release formulations supported by trials?",
        "Is dosing guidance changed by new evidence?",
        "Are there new safety signals (overdose, QTc, other)?",
        "Should pregnancy recommendations change for this drug?",
        "Should adolescent guidance change for this drug?",
        "Does the drug perform differently in co-occurring psychiatric disorders?",
        "Is transition between medications better informed?",
        "Does adding psychosocial interventions change outcomes with this drug?"
    ]
    custom_questions = st.text_area("Custom questions (one per line)", height=120)

# Build questions list
if use_default:
    questions = default_questions.copy()
else:
    questions = []

if custom_questions.strip():
    for line in custom_questions.splitlines():
        if line.strip():
            questions.append(line.strip())

# -----------------------------
# Reviewer route: check query params
# -----------------------------
params = st.experimental_get_query_params()
if "review_token" in params:
    token = params["review_token"][0] if isinstance(params["review_token"], list) else params["review_token"]
    reviews = load_json(REVIEW_FILE)
    if token not in reviews:
        st.error("Invalid or expired review token.")
        st.stop()

    review = reviews[token]
    st.header(f"Reviewer page — {review.get('drug')}")
    st.write("Status:", review.get("status"))
    st.write("Conflicts (summary):")
    st.json(review.get("conflicts"))

    st.subheader("Review decisions")
    reviewer_email = st.text_input("Your email (reviewer)")
    decision = st.radio("Decision", ["approve", "reject", "needs_changes"], index=0)
    notes = st.text_area("Notes for the team / rationale", height=200)

    if st.button("Submit review"):
        review["status"] = decision
        review["reviewer_email"] = reviewer_email
        review["notes"] = notes
        review["reviewed_at"] = now = __import__("datetime").datetime.utcnow().isoformat()
        # save back
        data = load_json(REVIEW_FILE)
        data[token] = review
        Path(REVIEW_FILE).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        st.success("Review saved. Thank you.")
    st.stop()

# -----------------------------
# Main UI: run pipeline
# -----------------------------
st.markdown("## Run pipeline")
drug = st.text_input("Drug name (e.g., Buprenorphine, Methadone, Naltrexone):")

col1, col2 = st.columns([3,1])
with col1:
    if st.button("Run Pipeline"):
        if not drug.strip():
            st.error("Enter a drug name.")
        else:
            # Progress UI elements
            status_text = st.empty()
            progress_bar = st.progress(0.0)

            def progress_callback(step_idx, message, progress_fraction):
                # progress_fraction expected 0.0 - 1.0
                try:
                    progress_bar.progress(min(max(float(progress_fraction), 0.0), 1.0))
                except Exception:
                    pass
                status_text.info(f"[{int(progress_fraction*100)}%] {message}")

            # call pipeline with progress callback
            result = run_pipeline_for_drug(drug, questions, progress_callback=progress_callback)
            st.success("Pipeline finished.")
            # debug: show pmids count
            st.write("PMIDs fetched:", len(result.get("pmids", [])))
            st.write("Saved to:", OUTDIR / f"pipeline_{(drug or 'unknown')}.json")

            # show summary safely
            summary = result.get("summary")
            if not summary:
                st.warning("No summary produced — showing raw result for debugging.")
                st.json(result)
            else:
                st.subheader("Summary")
                st.json(summary)

            # if conflicts -> show reviewer options (email)
            conflicts = result.get("conflicts", {})
            if conflicts.get("has_conflict"):
                st.warning("Conflicts detected between papers. Reviewer action recommended.")
                token = result.get("review_token")
                st.write("Review token:", token)

                # default reviewer email field
                reviewer_address = st.text_input("Reviewer email to notify (recipient):")

                if st.button("Send review email"):
                    if not reviewer_address:
                        st.error("Enter reviewer email address.")
                    elif not smtp_username or not smtp_password:
                        st.error("Enter SMTP username and password in the sidebar.")
                    else:
                        # construct review link; assume local host if running locally
                        base_url = st.experimental_get_query_params().get("base_url", [None])[0] if st.experimental_get_query_params().get("base_url") else None
                        # build a basic link — user may need to replace host with public URL when deployed
                        host = st.experimental_get_query_params().get("host", [None])[0] or "http://localhost:8501"
                        review_link = f"{host}/?review_token={token}"
                        subject = f"Review required: guideline update for {drug}"
                        body = f"Dear reviewer,\n\nA conflict was detected for drug '{drug}'. Please review: {review_link}\n\nToken: {token}\n\nThank you."
                        try:
                            send_review_email(smtp_server, smtp_port, smtp_username, smtp_password, smtp_username, reviewer_address, subject, body, sender_name=sender_name)
                            st.success("Review email sent.")
                        except Exception as e:
                            st.error(f"Failed to send email: {e}")

            # show per-question answers in an expander
            st.subheader("Per-question answers")
            for ans in result.get("results", []):
                q = ans.get("_question", "")
                with st.expander(q):
                    # hide the raw retrieved passages (too large)
                    safe_ans = {k:v for k,v in ans.items() if k not in ("_retrieved",)}
                    st.json(safe_ans)
                    retrieved = ans.get("_retrieved", [])
                    if retrieved:
                        st.write("Top retrieved evidence (PMID + snippet):")
                        for r in retrieved[:5]:
                            pm = r.get("meta", {}).get("pmid")
                            sample = (r.get("passage") or "")[:350].replace("\n"," ")
                            st.write(f"- PMID {pm}: {sample}...")

            # show classifications
            st.subheader("Paper classifications (stance)")
            for p in result.get("classifications", []):
                st.write(f"PMID: {p.get('pmid')} — stance: {p.get('stance')}")
                if p.get("summary"):
                    st.caption(p.get("summary"))
                st.markdown("---")

            # download button
            st.download_button("Download results JSON", data=json.dumps(result, indent=2, ensure_ascii=False), file_name=f"pipeline_{drug}.json")

# -----------------------------
# helper: send email function
# -----------------------------
def send_review_email(server, port, username, password, sender_email, recipient_email, subject, body, sender_name="Guideline Bot"):
    """
    Send email via SMTP. Supports SSL (port 465) and STARTTLS (port 587).
    For Gmail: use smtp.gmail.com and an App Password with port 465 or 587.
    """
    msg = EmailMessage()
    msg["From"] = f"{sender_name} <{sender_email}>"
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.set_content(body)

    context = ssl.create_default_context()
    port = int(port)
    if port == 465:
        with smtplib.SMTP_SSL(server, port, context=context) as smtp:
            smtp.login(username, password)
            smtp.send_message(msg)
    else:
        # try STARTTLS
        with smtplib.SMTP(server, port, timeout=60) as smtp:
            smtp.ehlo()
            smtp.starttls(context=context)
            smtp.ehlo()
            smtp.login(username, password)
            smtp.send_message(msg)
