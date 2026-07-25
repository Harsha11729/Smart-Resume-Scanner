# 📄 Smart Resume Scanner

An AI-powered Resume Scanner and Analyzer built using **Python**, **Streamlit**, **spaCy**, and **MySQL**. The application extracts candidate information from PDF resumes, detects technical skills, compares resumes with job descriptions using cosine similarity, and stores the analysis results in a database through an interactive web interface.

---

## 🚀 Features

- 📂 Upload resumes in PDF format
- 📑 Extract candidate information:
  - Name
  - Email
  - Phone Number
- 🛠 Detect technical skills using NLP (spaCy PhraseMatcher)
- 🤖 Confidence-based AI skill prediction module
- 🎯 Resume and Job Description (JD) matching
- 📊 Resume score using TF-IDF & Cosine Similarity
- 📈 Skill distribution visualization
- 💾 Store analysis results in MySQL database
- 🔐 Admin Dashboard
  - View uploaded resume details
  - Download data as CSV
  - Clear database records
- 📄 Preview uploaded resume inside the application

---

## 🖥️ Tech Stack

### Frontend
- Streamlit

### Backend
- Python

### Database
- MySQL

### NLP & Machine Learning
- spaCy
- scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity
- PyTorch (Confidence Prediction Module)

### Libraries
- pdfplumber
- pandas
- matplotlib
- seaborn
- Pillow
- PyMySQL

---

## 📂 Project Structure

```
Smart-Resume-Scanner/
│── app.py
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/Smart-Resume-Scanner.git
cd Smart-Resume-Scanner
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download the spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### Configure MySQL

1. Install MySQL.
2. Create a database named `cv`.
3. Update your MySQL username and password in `app.py`.

### Run the Application

```bash
streamlit run app.py
```

---

## 📊 Workflow

1. Upload a Resume (PDF).
2. (Optional) Upload a Job Description.
3. The system extracts candidate information.
4. Skills are detected using NLP.
5. Resume similarity with the Job Description is calculated.
6. Resume score and detected skills are displayed.
7. Results are stored in the MySQL database.
8. Admin can view, download, or manage stored records.

---

## 📸 Screenshots

Add screenshots of:
- Home Page
- Resume Upload
- Resume Analysis
- Skill Visualization
- Admin Dashboard

---

## 🔮 Future Enhancements

- ATS Compatibility Score
- Resume Ranking
- Resume Recommendation System
- AI Resume Improvement Suggestions
- Cover Letter Generator
- Interview Question Generator
- Multi-language Resume Support
- Cloud Database Integration
- Recruiter Dashboard

---

## 👨‍💻 Author

**Harsha Kandepu**

B.Tech – Computer Science & Engineering

---

## ⭐ Support

If you found this project helpful, consider giving it a **⭐ Star** on GitHub!
