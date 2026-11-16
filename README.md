# AI Resume Ranker (ATS Score Checker)

Simple AI-powered tool that calculates how well your resume matches a job description using TF-IDF + Cosine Similarity. Perfect for AI Intern portfolio.

## Features
- Upload PDF resume or paste text
- Paste job description
- Get ATS Match Score (%)
- See missing important keywords
- Pure Python + scikit-learn (no heavy models)

## How to Run (Google Colab - Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Create new notebook
3. Copy the entire code from ``resume_ranker.py` + **`AI_Resume_Ranker.ipynb`**` (or run the cells)
4. Upload your resume PDF when prompted
5. Paste the job description
6. See your score + missing keywords

You can also run the `.py` file locally after installing requirements.

## Installation (Local)

```bash
pip install -r requirements.txt
python `resume_ranker.py` + **`AI_Resume_Ranker.ipynb`**
```

## Tech Stack
- Python
- scikit-learn (TF-IDF + Cosine Similarity)
- NLTK
- PyPDF2

## Project Structure
```
01-ai-resume-ranker/
├── `resume_ranker.py` + **`AI_Resume_Ranker.ipynb`**
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Why this project?
Shows practical NLP skills useful for AI Intern roles. Recruiters love seeing real-world tools.

## Author
Made for AI Intern portfolio. Feel free to fork and improve.
