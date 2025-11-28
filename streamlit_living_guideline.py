# streamlit_living_guideline.py
import streamlit as st
import json
import smtplib, ssl
from email.message import EmailMessage
from pathlib import Path
from pipeline_code import run_pipeline_for_drug, set_openai_key, load_json, REVIEW_FILE, OUTDIR

st.set_page_config(page_title="ASAM Updater (Reviewer B + S3 + F2)", layout="wide")
st.title("ASAM Living Guideline Updater — Reviewer B / Summary S3 / Visibility F2")
st.caption("LLM answers visible immediately (NOT REVIEWED). Reviewer approves entire set and edits AI summary.")

# ---------------------------
# Sidebar: credentials & settings
# ---------------------------
with st.sidebar:
    st.header("Settings & Credentials")
    api_key = st.text_input("OpenAI API Key (optional)", type="password")
    if st.button("Set OpenAI Key"):
        set_openai_key(api_key)
        st.success("OpenAI key set (or simulation mode if blank).")

    st.markdown("---")
    st.subheader("SMTP (for review emails)")
    smtp_server = st.text_input("SMTP server", value="smtp.gmail.com")
    smtp_port = st.number_input("SMTP port", value=465)
    smtp_username = st.text_input("SMTP username (sender email)")
    smtp_password = st.text_input("SMTP password or app password", type="password")
    sender_name = st.text_input("Sender name", value="Guideline Bot")

    st.markdown("---")
    st.subheader("Default reviewer email (optional)")
    reviewer_default = st.text_input("Reviewer email (default recipient)")

    st.markdown("---")
    st.subheader("Questions")
    default_q = st.checkbox("Use default 10 questions", value=True)
    custom_q = st.text_area("Custom questions (one per line)", height=120)
    if default_q:
        questions = [
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
    else:
        questions = []
    if custom_q.strip():
        for line in custom_q.splitlines():
            if line.strip():
                questions.append(line.strip())

# ---------------------------
# Helper: send review email
# ---------------------------
def send_review_email(server, port, username, password, sender_email, recipient_email, subject, body, sender_name="Guideline Bot"):
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
        with smtplib.SMTP(server, port, timeout=60) as smtp:
            smtp.ehlo()
            smtp.starttls(context=context)
            smtp.ehlo()
            smtp.login(username, password)
            smtp.send_message(msg)

# ---------------------------
# Run pipeline section
# ---------------------------
st.markdown("## Run pipeline")
drug = st.text_input("Drug name (e.g., Buprenorphine, Methadone, Naltrexone)")

col1, col2 = st.columns([3,1])
with col1:
    if st.button("Run Pipeline"):
        if not drug.strip():
            st.error("Enter a drug name")
        else:
            status = st.empty()
            pbar = st.progress(0.0)
            def progress_cb(step, msg, frac):
                try:
                    pbar.progress(min(max(float(frac),0.0),1.0))
                except Exception:
                    pass
                status.info(f"[{int(frac*100)}%] {msg}")

            result = run_pipeline_for_drug(drug, questions, progress_callback=progress_cb)
            st.success("Pipeline finished")
            st.write("PMIDs fetched:", len(result.get("pmids",[])))
            st.write("Review token (if created):", result.get("review_token"))

            # Save local reference path
            saved_path = OUTDIR / f"pipeline_{drug or 'unknown'}.json"
            st.write("Saved:", saved_path)

            # Show AI answers immediately — labeled NOT REVIEWED
            st.subheader("AI per-question answers (NOT REVIEWED unless reviewed later)")
            for ans in result.get("results", []):
                q = ans.get("_question","")
                with st.expander(q):
                    # show core LLM fields
                    st.write("Recommendation:", ans.get("recommendation_text"))
                    st.write("Strength:", ans.get("strength"))
                    st.write("Rationale:", ans.get("rationale"))
                    st.write("Evidence used (PMIDs):", ans.get("evidence_used"))
                    st.info("Status: NOT REVIEWED")

            # Show AI-generated summary (editable by reviewer later)
            st.subheader("AI-generated summary (editable by reviewer)")
            ai_summary = result.get("summary", {}).get("ai_summary")
            if ai_summary:
                st.markdown(ai_summary)
            else:
                st.info("No AI summary produced.")

            # If review token exists, allow sending review email
            if result.get("review_token"):
                st.warning("A review token was created (conflict or updates suggested).")
                token = result.get("review_token")
                # recipient: default reviewer or field show
                reviewer_addr = reviewer_default or st.text_input("Reviewer email to notify", key=f"rev_{token}")
                if st.button("Send review email", key=f"send_{token}"):
                    if not reviewer_addr:
                        st.error("Enter reviewer email")
                    elif not smtp_username or not smtp_password:
                        st.error("Fill SMTP username & password in sidebar")
                    else:
                        # build review link (must be public if deployed)
                        base = st.experimental_get_query_params().get("base", [None])[0] or "http://localhost:8501"
                        review_link = f"{base}/?review_token={token}"
                        subj = f"Review request: {drug} guideline update"
                        body = f"Please review/update the AI summary and accept/reject the suggested updates.\n\nReview link: {review_link}\nToken: {token}"
                        try:
                            send_review_email(smtp_server, smtp_port, smtp_username, smtp_password, smtp_username, reviewer_addr, subj, body, sender_name=sender_name)
                            st.success("Review email sent.")
                        except Exception as e:
                            st.error(f"Email failed: {e}")

# ---------------------------
# Reviewer route: ?review_token=...
# This page implements Option B (single final decision) + S3 (AI summary editable)
# ---------------------------
params = st.experimental_get_query_params()
if "review_token" in params:
    token = params["review_token"][0] if isinstance(params["review_token"], list) else params["review_token"]
    reviews = load_json(REVIEW_FILE)
    if token not in reviews:
        st.error("Invalid or expired review token.")
        st.stop()

    rev = reviews[token]
    st.header(f"Reviewer page — {rev.get('drug')}")
    st.write("Created at:", rev.get("created_at"))
    st.subheader("Conflicts (overview)")
    st.json(rev.get("conflicts", {}))
    st.subheader("AI per-question recommendations (read-only)")
    for item in rev.get("results", []):
        q = item.get("_question", "")
        with st.expander(q):
            st.write("AI recommendation:", item.get("recommendation_text"))
            st.write("Strength:", item.get("strength"))
            st.write("Rationale:", item.get("rationale"))
            st.write("Evidence (PMIDs):", item.get("evidence_used"))

    st.subheader("AI-generated summary (editable)")
    ai_sum = rev.get("ai_summary") or ""
    edited = st.text_area("Edit AI summary (1-2 paragraphs)", value=ai_sum, height=200)

    st.subheader("Reviewer decision (single overall decision)")
    reviewer_email = st.text_input("Your email (for record)")
    decision = st.radio("Decision", ["approve", "reject", "needs_changes"], index=0)
    notes = st.text_area("Notes (optional)", height=150)

    if st.button("Submit reviewer decision"):
        reviews[token]["reviewer_summary"] = edited.strip()
        reviews[token]["reviewer_email"] = reviewer_email
        reviews[token]["reviewer_decision"] = decision
        reviews[token]["reviewer_notes"] = notes
        reviews[token]["status"] = "approved" if decision == "approve" else ("rejected" if decision == "reject" else "needs_changes")
        reviews[token]["reviewed_at"] = datetime.utcnow().isoformat()
        Path(REVIEW_FILE).write_text(json.dumps(reviews, indent=2, ensure_ascii=False), encoding="utf-8")
        st.success("Reviewer decision saved. Thank you.")
        st.experimental_rerun()
    st.stop()

# ---------------------------
# Display reviewed decisions for a drug if present (optional quick lookup)
# ---------------------------
st.markdown("---")
st.header("Lookup review status (optional)")
lookup_token = st.text_input("Enter review token to view status (optional)")
if st.button("Lookup token"):
    data = load_json(REVIEW_FILE)
    if lookup_token in data:
        st.json(data[lookup_token])
    else:
        st.warning("Token not found.")
