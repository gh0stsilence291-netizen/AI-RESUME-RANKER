"""
AI Resume Ranker - ATS Score Checker
Simple TF-IDF based resume vs job description matcher.
Works in Google Colab and locally.
"""

import re
import sys

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Install requirements first: pip install -r requirements.txt")
    sys.exit(1)

try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    stop_words = set(stopwords.words('english'))
except:
    stop_words = set()

try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False


def clean_text(text: str) -> str:
    """Basic text cleaning for TF-IDF."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF resume."""
    if not HAS_PDF:
        return ""
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"PDF read error: {e}")
    return text


def calculate_score(resume_text: str, job_description: str):
    """Calculate cosine similarity score between resume and JD."""
    clean_resume = clean_text(resume_text)
    clean_jd = clean_text(job_description)

    if not clean_resume or not clean_jd:
        return 0.0, set()

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([clean_jd, clean_resume])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    jd_words = set(clean_jd.split())
    resume_words = set(clean_resume.split())
    missing = jd_words - resume_words

    return score, missing


def main():
    print("=" * 55)
    print("   AI Resume Ranker - ATS Score Checker")
    print("=" * 55)
    print()

    # Try Colab file upload first
    resume_text = ""
    try:
        from google.colab import files
        print("Colab detected. Upload your Resume PDF (or cancel to paste text)...")
        uploaded = files.upload()
        if uploaded:
            for filename in uploaded.keys():
                resume_text = extract_text_from_pdf(filename)
                print(f"Extracted text from: {filename}")
                break
    except ImportError:
        # Local mode
        pdf_path = input("Enter path to resume PDF (or press Enter to paste text): ").strip()
        if pdf_path:
            resume_text = extract_text_from_pdf(pdf_path)

    if not resume_text:
        print("\nPaste your full resume text below (end with empty line or Ctrl+D):")
        lines = []
        try:
            while True:
                line = input()
                if not line.strip():
                    break
                lines.append(line)
        except EOFError:
            pass
        resume_text = "\n".join(lines)

    print("\nPaste the Job Description below (end with empty line):")
    lines = []
    try:
        while True:
            line = input()
            if not line.strip():
                break
            lines.append(line)
    except EOFError:
        pass
    job_description = "\n".join(lines)

    if not resume_text.strip() or not job_description.strip():
        print("Resume or Job Description empty. Exiting.")
        return

    score, missing = calculate_score(resume_text, job_description)

    print("\n" + "=" * 55)
    print(f"  ATS MATCH SCORE : {round(score * 100, 2)}%")
    print("=" * 55)

    if missing:
        print("\nMissing important keywords (top 20):")
        print(", ".join(list(missing)[:20]))
    else:
        print("\nNo major keywords missing. Strong match!")

    print("\nTip: Add the missing keywords naturally in your resume to improve score.")
    print("=" * 55)


if __name__ == "__main__":
    main()
