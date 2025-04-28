import streamlit as st
from utils import extract_text_from_pdf, clean_text
from ranker import rank_resumes

st.title("NLP-Based Resume Ranker")

job_description = st.text_area("Paste the Job Description")

resume_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

if st.button("Rank Resumes"):
    if not job_description or not resume_files:
        st.warning("Please provide both Job Description and Resume files.")
    else:
        jd_clean = clean_text(job_description)
        resume_texts = [clean_text(extract_text_from_pdf(resume)) for resume in resume_files]
        ranked = rank_resumes(resume_texts, jd_clean)

        st.subheader("Ranking Results:")
        for index, score in ranked:
            st.write(f"{resume_files[index].name} - Match Score: {score}")
