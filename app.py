import streamlit as st
import pandas as pd
import base64
import datetime
import os
import io
import re
import pdfplumber
from PIL import Image
import pymysql
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.matcher import PhraseMatcher
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import random # Ensure random is imported
import warnings # For warning handling

# Suppress the specific FutureWarning from seaborn if needed, though the fix below addresses it
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# --- Mock EDLNet and common_skills for demonstration if you don't have them set up ---
# IMPORTANT: Replace these mock implementations with your actual EDLNet model and prediction logic.
# These mocks are for demonstration purposes only and will affect the accuracy of skill detection.

# Define a comprehensive list of common skills
common_skills = [
    # Programming Languages & Frameworks
    "Python", "Java", "C++", "JavaScript", "TypeScript", "Go", "Rust", "PHP", "Ruby", "Swift", "Kotlin", "C#",
    "HTML", "CSS", "SQL", "NoSQL", "Bash", "Shell Scripting", "R", "Scala", "Perl", "VBA", "MATLAB",
    "Mathematica", "Maple", # Added based on user feedback
    "LaTeX", "Sphinx", # Added based on user feedback

    # Web Development
    "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Spring Boot", "ASP.NET", "Ruby on Rails",
    "Express.js", "jQuery", "Bootstrap", "Tailwind CSS", "Sass", "Less", "REST API", "GraphQL", "Microservices",
    "Frontend Development", "Backend Development", "Full Stack Development", "WebSockets",

    # Mobile Development
    "Android Development", "iOS Development", "React Native", "Flutter", "Xamarin", "SwiftUI", "Kotlin Multiplatform",

    # Cloud Platforms
    "AWS", "Azure", "Google Cloud Platform", "Kubernetes", "Docker", "Terraform", "Ansible", "CloudFormation",
    "Serverless", "Lambda", "EC2", "S3", "RDS", "VPC", "Azure DevOps", "Google Kubernetes Engine", "OpenStack",
    "Heroku", "Netlify", "Vercel",

    # Databases
    "MySQL", "PostgreSQL", "MongoDB", "Cassandra", "Redis", "Elasticsearch", "Oracle", "SQL Server", "SQLite",
    "Firebase", "DynamoDB", "Neo4j", "Apache Kafka", "RabbitMQ", "Data Lake", "Data Warehouse",

    # Data Science & Machine Learning
    "Machine Learning", "Deep Learning", "Data Science", "Artificial Intelligence", "Natural Language Processing",
    "Computer Vision", "Reinforcement Learning", "Neural Networks", "Pandas", "NumPy", "Scikit-learn",
    "TensorFlow", "PyTorch", "Keras", "Apache Spark", "Hadoop", "Kafka", "Data Analysis", "Statistical Analysis",
    "Big Data", "Data Visualization", "Matplotlib", "Seaborn", "Plotly", "Dash", "Streamlit", "Power BI", "Tableau",
    "ETL", "Data Warehousing", "Data Modeling", "A/B Testing", "Feature Engineering", "Model Deployment",
    "Predictive Modeling", "Econometrics", "Bioinformatics", "Genomics", "Time Series Analysis",
    "Data Mining", # Added based on user feedback

    # DevOps & CI/CD
    "Git", "GitHub", "GitLab", "Bitbucket", "Jenkins", "CircleCI", "Travis CI", "Jira", "Confluence", "Agile", "Scrum",
    "Kanban", "CI/CD", "Configuration Management", "Monitoring", "Logging", "Prometheus", "Grafana", "Splunk",
    "ELK Stack", "Ansible", "Chef", "Puppet", "Nagios", "Vagrant", "Nexus", "Artifactory",
    "CVS", "HTCondor", # Added based on user feedback

    # Operating Systems & Virtualization
    "Linux", "Ubuntu", "CentOS", "Red Hat", "Windows Server", "VMware", "VirtualBox", "Hyper-V", "Unix",

    # Cybersecurity
    "Cybersecurity", "Network Security", "Information Security", "Penetration Testing", "Vulnerability Assessment",
    "Firewalls", "VPN", "Encryption", "SIEM", "Incident Response", "Compliance", "GDPR", "HIPAA", "ISO 27001",
    "Threat Intelligence", "Malware Analysis", "Security Audits", "Risk Assessment",

    # Project Management & Business Skills
    "Project Management", "Product Management", "Scrum Master", "Agile Methodologies", "Stakeholder Management",
    "Risk Management", "Budgeting", "Resource Allocation", "Communication", "Teamwork", "Leadership",
    "Problem Solving", "Critical Thinking", "Adaptability", "Time Management", "Negotiation", "Presentation Skills",
    "Strategic Planning", "Business Analysis", "Client Relations", "Customer Service", "Emotional Intelligence",
    "Conflict Resolution", "Decision Making", "Mentoring", "Coaching", "Report Writing", "Documentation",
    "Market Research", "Financial Modeling", "Public Speaking", "Operations Management", "Supply Chain Management",
    "Quality Management", "Change Management", "Process Improvement", "Six Sigma", "Lean Manufacturing",

    # Design & UX/UI
    "UI/UX Design", "Figma", "Sketch", "Adobe XD", "Photoshop", "Illustrator", "InDesign", "Wireframing",
    "Prototyping", "User Research", "Usability Testing", "Information Architecture", "Interaction Design",
    "Visual Design", "Graphic Design", "Motion Graphics", "Branding", "Accessibility",

    # Quality Assurance & Testing
    "Quality Assurance", "Software Testing", "Manual Testing", "Automation Testing", "Selenium", "Cypress",
    "JMeter", "Test Cases", "Test Plans", "Bug Tracking", "Performance Testing", "Regression Testing",
    "User Acceptance Testing", "Test Automation Frameworks",

    # Other Tools & Concepts
    "Microsoft Office", "Google Suite", "Slack", "Zoom", "CRM", "ERP", "Salesforce", "Blockchain", "IoT",
    "AR/VR", "Quantum Computing", "Robotics", "GIS", "CAD", "SAP", "ServiceNow", "Zendesk", "HubSpot",
    "Tableau Desktop", "Power BI Desktop", "Confluence", "SharePoint", "Outlook", "Excel", "Word", "PowerPoint"
]

# Aliases/Synonyms for skills (map to canonical form)
skill_aliases = {
    "ml": "Machine Learning",
    "ai": "Artificial Intelligence",
    "nlp": "Natural Language Processing",
    "cv": "Computer Vision",
    "k8s": "Kubernetes",
    "js": "JavaScript",
    "ts": "TypeScript",
    "gcp": "Google Cloud Platform",
    "aws": "AWS", # Keep canonical form if it's already there
    "az": "Azure",
    "data analysis": "Data Analysis", # Ensure canonical forms are consistently mapped
    "data analytics": "Data Analysis",
    "data mining": "Data Mining", # Added alias for consistency
    "project mgmt": "Project Management",
    "pm": "Project Management",
    "product mgmt": "Product Management",
    "qa": "Quality Assurance",
    "ci/cd": "CI/CD",
    "nosql": "NoSQL",
    "rest": "REST API", # Map "REST" to "REST API"
    "cyber sec": "Cybersecurity",
    "info sec": "Information Security"
}

# --- Mock EDLNet and predict_skills_with_edl function ---
class MockEDLNet:
    def __init__(self):
        pass
    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        log_probs = torch.rand(batch_size, seq_len, len(common_skills)).log_softmax(dim=-1)
        total_evidence = torch.rand(batch_size, seq_len, len(common_skills))
        return log_probs, total_evidence

def predict_skills_with_edl(text_chunk):
    """
    Mock function to simulate EDLNet skill prediction.
    IMPORTANT: This mock is a keyword-based simulation and cannot replicate the nuanced
    understanding of a real, trained EDLNet model. This version is heavily tuned
    to avoid false positives for "Data Science" based on general "Python" and "analysis"
    in a cybersecurity context.
    """
    predicted_skills_with_confidence = []
    text_lower = text_chunk.lower()

    # Define a subset of common_skills that are "controlled" for random prediction
    # This prevents broad terms like "Data Science" from being randomly added by this loop.
    controlled_common_skills = [s for s in common_skills if s.lower() not in [
        "data science", "machine learning", "artificial intelligence", "deep learning",
        "natural language processing", "computer vision", "predictive modeling",
        "data modeling", "feature engineering", "data analysis", "statistical analysis",
        "big data", "devops", "cybersecurity", "project management", # Exclude these broad terms from random pool
        "communication", "leadership", "teamwork", "problem solving", "critical thinking", # Also exclude common soft skills from random generation
        "microsoft office", "google suite", "slack", "zoom" # Exclude common office tools from random generation
    ]]

    # Simulate some highly confident direct matches if very specific keywords are present
    # These are skills that if present, are very likely to be actual skills.
    direct_match_candidates = [
        "Python", "Java", "C++", "JavaScript", "SQL", "React", "Angular", "Vue.js", "Node.js",
        "AWS", "Azure", "Google Cloud Platform", "Kubernetes", "Docker", "Git", "Terraform",
        "Firewalls", "SIEM", "Penetration Testing", "Vulnerability Assessment", "Incident Response",
        "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Keras", "Apache Spark",
        "Jupyter", "Matplotlib", "Seaborn", "Tableau", "Power BI", "MySQL", "PostgreSQL", "MongoDB",
        "Agile", "Scrum", "Jira", "Jenkins", "Ansible", "Linux", "Windows Server", "Figma", "Adobe XD",
        "Selenium", "Cypress", "JMeter", "Blockchain", "IoT",
        # Added based on user feedback for direct high confidence detection
        "Data Mining", "Data Analysis", "Sphinx", "LaTeX", "Mathematica", "Maple", "CVS", "HTCondor"
    ]

    for skill in direct_match_candidates:
        if skill.lower() in text_lower:
            # Assign very high confidence for these specific matches
            predicted_skills_with_confidence.append((skill, random.uniform(0.90, 0.99)))

    # --- Explicit Inference Logic for Broad/Complex Skills ---
    # These rules are designed to be stricter and require multiple indicators for high confidence.

    # Data Science / Machine Learning / AI Inference - EXTREMELY STRICT
    infer_data_science_ml_ai = False
    ds_keywords = ["data science", "machine learning", "artificial intelligence", "deep learning", "nlp", "computer vision", "predictive modeling", "data mining", "data analysis"] # Added data mining/analysis here
    ds_libs = ["pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "jupyter"]
    ds_concepts = ["statistical analysis", "data modeling", "feature engineering", "algorithm development", "neural networks"]

    # Rule 1: Strong explicit direct mentions of the broad field
    if any(term in text_lower for term in ds_keywords):
        infer_data_science_ml_ai = True
    
    # Rule 2: Presence of Python/R AND specific, unmistakable DS/ML libraries/concepts
    if "python" in text_lower and any(lib in text_lower for lib in ds_libs):
        infer_data_science_ml_ai = True
    if "r" in text_lower and any(term in text_lower for term in ["statistical analysis", "ggplot2", "dplyr", "caret"]):
        infer_data_science_ml_ai = True
    
    # Rule 3: Big Data technologies specifically with data processing terms
    if ("apache spark" in text_lower or "hadoop" in text_lower) and ("data processing" in text_lower or "big data" in text_lower or "data engineering" in text_lower):
        infer_data_science_ml_ai = True
    
    # Add if inferred with high confidence
    if infer_data_science_ml_ai:
        if not any(s[0] == "Data Science" for s in predicted_skills_with_confidence) and ("data science" in text_lower or ( "python" in text_lower and any(lib in text_lower for lib in ds_libs))):
            predicted_skills_with_confidence.append(("Data Science", random.uniform(0.85, 0.95))) # High confidence if inferred via strong signals
        if not any(s[0] == "Machine Learning" for s in predicted_skills_with_confidence) and "machine learning" in text_lower:
             predicted_skills_with_confidence.append(("Machine Learning", random.uniform(0.88, 0.96)))
        if not any(s[0] == "Artificial Intelligence" for s in predicted_skills_with_confidence) and "artificial intelligence" in text_lower:
             predicted_skills_with_confidence.append(("Artificial Intelligence", random.uniform(0.88, 0.96)))
        # Specifically for Data Mining/Analysis, ensure they are added if inferred strongly or explicitly mentioned
        if "data mining" in text_lower and not any(s[0] == "Data Mining" for s in predicted_skills_with_confidence):
            predicted_skills_with_confidence.append(("Data Mining", random.uniform(0.85, 0.95)))
        if "data analysis" in text_lower and not any(s[0] == "Data Analysis" for s in predicted_skills_with_confidence):
            predicted_skills_with_confidence.append(("Data Analysis", random.uniform(0.85, 0.95)))


    # Cybersecurity Inference
    cyber_keywords = ["cybersecurity", "network security", "information security", "threat intelligence",
                      "penetration testing", "vulnerability assessment", "incident response", "malware analysis",
                      "siem", "firewall", "vpn", "ids", "ips", "security operations", "digital forensics",
                      "compliance", "gdpr", "hipaa", "iso 27001", "threat intelligence", "security audits", "risk assessment"]
    if any(term in text_lower for term in cyber_keywords):
        if not any(s[0] == "Cybersecurity" for s in predicted_skills_with_confidence):
            predicted_skills_with_confidence.append(("Cybersecurity", random.uniform(0.85, 0.97)))

    # DevOps/CI/CD Inference
    devops_keywords = ["devops", "ci/cd", "continuous integration", "continuous delivery",
                       "jenkins", "gitlab ci", "github actions", "circleci", "travis ci",
                       "ansible", "chef", "puppet", "terraform", "cloudformation",
                       "docker", "kubernetes", # Cloud presence strongly implies DevOps practices
                       "monitoring", "logging", "prometheus", "grafana", "microservices", "serverless"]
    if any(term in text_lower for term in devops_keywords):
        if not any(s[0] == "DevOps" for s in predicted_skills_with_confidence):
            predicted_skills_with_confidence.append(("DevOps", random.uniform(0.85, 0.97)))

    # Project Management Inference
    pm_keywords = ["project management", "agile", "scrum", "kanban", "jira", "confluence",
                   "product management", "stakeholder management", "risk management", "budgeting",
                   "sprint planning", "retrospectives", "user stories", "gantt chart", "pmp"]
    if any(term in text_lower for term in pm_keywords):
        if not any(s[0] == "Project Management" for s in predicted_skills_with_confidence):
            predicted_skills_with_confidence.append(("Project Management", random.uniform(0.85, 0.97)))

    # --- General Random Skill Generation (for less prominent skills) ---
    # This loop uses the controlled list to avoid interfering with explicit broad skill inference
    for _ in range(random.randint(3, 10)): # Reduce the number of randomly added skills
        skill = random.choice(controlled_common_skills) # Use controlled list
        # Only add if not already predicted (especially by direct matches)
        if skill not in [s[0] for s in predicted_skills_with_confidence]:
            predicted_skills_with_confidence.append((skill, random.uniform(0.3, 0.85))) # Adjust confidence range

    # Final filter to ensure all skills have at least a moderate confidence
    predicted_skills_with_confidence = [(s, c) for s, c in predicted_skills_with_confidence if c >= 0.40]

    uncertainty = random.uniform(00.01, 0.4) # Simulate overall uncertainty, slightly lower max
    return predicted_skills_with_confidence, uncertainty
# --- End Mock EDLNet and predict_skills_with_edl function ---


# Load spaCy model
# Ensure 'en_core_web_sm' is downloaded: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    st.stop() # Stop the app if model is missing

matcher = PhraseMatcher(nlp.vocab)
# Add common skills as patterns to the matcher. Using unique string IDs.
# Add canonical forms and aliases
for skill in common_skills:
    matcher.add(skill.replace(" ", "_").upper(), [nlp.make_doc(skill)])
for alias, canonical_skill in skill_aliases.items():
    # Ensure aliases are added with a distinct ID to avoid clashes if an alias is also a canonical skill
    matcher.add(alias.replace(" ", "_").upper() + "_ALIAS", [nlp.make_doc(alias)])


# Connect to DB
connection = None
cursor = None
try:
    connection = pymysql.connect(host='localhost', user='root', password=__Your_password__, db='cv') # <--- Update MySQL Password
    cursor = connection.cursor()
    DB_table_name = 'user_data'
    cursor.execute(f"""
        CREATE DATABASE IF NOT EXISTS cv;
    """)
    cursor.execute(f"""
        USE cv;
    """)
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {DB_table_name} (
            ID INT AUTO_INCREMENT PRIMARY KEY,
            Name VARCHAR(255),
            Email_ID VARCHAR(255),
            Phone VARCHAR(20),
            resume_score DECIMAL(5,2),
            Timestamp DATETIME,
            Page_no VARCHAR(10),
            Skills TEXT
        )
    """)
    connection.commit()
except pymysql.Error as e:
    st.error(f"Error connecting to database: {e}. Please ensure MySQL is running and credentials are correct.")
    connection = None
    cursor = None

def extract_text_from_pdf(file):
    """Extracts text from a PDF file using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text(x_tolerance=2, y_tolerance=2) or ''
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text

def extract_info(text):
    """Extracts name, email, and phone number from the resume text with enhanced precision."""
    doc = nlp(text)
    name, email, phone = "Unknown", "Unknown", "Unknown"

    # --- Email Extraction ---
    email_match = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    if email_match:
        email = email_match[0]

    # --- Phone Number Extraction ---
    # Enhanced regex to capture more formats, including those with country codes or just digits
    # Prioritize longer sequences of digits, and common separators
    # FIXED: Concatenate multi-line regex into a single string.
    phone_match = re.findall(
        r"(?:(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)|" # International/US formats
        r"(?:\d{2,5}[-.\s]?){2,4}\d{2,5}(?!\d)|" # More general pattern for N-digit groups
        r"\b\d{7,15}\b)" # Just sequence of 7-15 digits. Added closing parenthesis for the group.
    , text)
    if phone_match:
        # Filter for plausible phone numbers (7-15 digits after cleaning)
        cleaned_phones = []
        for p in phone_match:
            digits = re.sub(r'\D', '', p)
            if 7 <= len(digits) <= 15: # Standard phone number lengths
                cleaned_phones.append(digits)
        if cleaned_phones:
            # Prioritize phone numbers appearing earlier in the text
            phone = sorted(cleaned_phones, key=lambda x: text.find(x))[0]

    # --- Name Extraction (Highly Refined Strategy for Full Name) ---
    potential_names = []
    text_lower = text.lower()
    
    # Keywords that indicate a non-name heading/section
    non_name_indicators = [
        "resume", "curriculum vitae", "contact", "profile", "summary", "objective",
        "experience", "education", "skills", "projects", "portfolio", "about me",
        "professional", "developer", "engineer", "analyst", "manager", "specialist",
        "certifications", "awards", "publications", "work history", "employment",
        "contact information", "technical skills", "honors", "references", "languages",
        "career", "goals", "introduction", "personal details", "address", "phone", "email",
        "github", "linkedin", "website"
    ]
    
    # 1. Look for name near contact info (most reliable)
    contact_info_segment = ""
    contact_indices = []
    if email != "Unknown":
        contact_indices.append(text_lower.find(email.lower()))
    if phone != "Unknown":
        contact_indices.append(text_lower.find(re.sub(r'\D', '', phone).lower()))
    
    if contact_indices:
        # Define a window around the earliest contact info
        earliest_contact_idx = min(idx for idx in contact_indices if idx != -1)
        if earliest_contact_idx != -1:
            search_start = max(0, earliest_contact_idx - 250) # Look back up to 250 chars
            search_end = min(len(text), earliest_contact_idx + 100) # Look forward up to 100 chars
            contact_info_segment = text[search_start:search_end]
            
            contact_doc = nlp(contact_info_segment)
            for ent in contact_doc.ents:
                if ent.label_ == "PERSON":
                    candidate_name = ent.text.strip()
                    # Basic validation for name: 2-5 words, not too short, not a common non-name keyword
                    if 2 <= len(candidate_name.split()) <= 5 and len(candidate_name) > 3:
                        # Ensure it's not all uppercase unless very short (e.g., "DR. JOHN DOE")
                        if not (candidate_name.isupper() and len(candidate_name.split()) > 1 and len(candidate_name) > 5):
                            if not any(indicator in candidate_name.lower() for indicator in non_name_indicators):
                                potential_names.append((candidate_name, "contact_proximity", contact_info_segment.find(candidate_name)))
            
            # If a strong candidate is found near contact, prioritize it
            if potential_names:
                potential_names.sort(key=lambda x: x[2]) # Sort by appearance in segment
                name = potential_names[0][0]
                return name, email, phone

    # 2. Analyze the very top lines for prominent capitalized names (if not found by contact proximity)
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    top_text_for_name = " ".join(lines[:5]) # Consider first 5 non-empty lines
    doc_top = nlp(top_text_for_name)

    # Look for spaCy PERSON entities in the top segment
    for ent in doc_top.ents:
        if ent.label_ == "PERSON":
            candidate_name = ent.text.strip()
            if 2 <= len(candidate_name.split()) <= 5 and len(candidate_name) > 3:
                if not (candidate_name.isupper() and len(candidate_name.split()) > 1 and len(candidate_name) > 5):
                    if not any(indicator in candidate_name.lower() for indicator in non_name_indicators):
                        potential_names.append((candidate_name, "spacy_person_top", top_text_for_name.find(candidate_name)))

    # Look for capitalized sequences (e.g., "John Doe", "Mary Ann Smith") using regex
    # This pattern specifically looks for words starting with a capital letter followed by lowercase,
    # ensuring it's not just "IBM" or "USA". It allows for 2 to 5 such words.
    # Added a lookbehind to avoid capturing titles like "Dr. John Doe" as just "John Doe" initially if Dr. is part of the name
    name_pattern_re = re.compile(r'\b(?:Mr\.|Ms\.|Dr\.|Engr\.|Prof\.)?\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,4})\b')
    matches = name_pattern_re.findall(top_text_for_name)
    
    for match_group in matches:
        candidate_name = match_group.strip()
        if 2 <= len(candidate_name.split()) <= 5 and len(candidate_name) > 3:
            if not (candidate_name.isupper() and len(candidate_name.split()) > 1 and len(candidate_name) > 5):
                if not any(indicator in candidate_name.lower() for indicator in non_name_indicators):
                    potential_names.append((candidate_name, "top_capitalized", top_text_for_name.find(candidate_name)))

    # Filter and select the best name from all potential candidates found so far
    if potential_names:
        # Sort by type preference (contact_proximity > spacy_person_top > top_capitalized)
        # and then by appearance order in the *original* text
        potential_names.sort(key=lambda x: (
            0 if x[1] == "contact_proximity" else
            1 if x[1] == "spacy_person_top" else
            2 if x[1] == "top_capitalized" else
            3, # Fallback for any other type if added
            text.find(x[0]) # Use original text for earliest appearance
        ))
        name = potential_names[0][0] # Take the most preferred candidate

    # Step 3: Final Fallback - If still "Unknown", consider a very broad search (less precise)
    if name == "Unknown":
        # Look for the first PERSON entity in the entire document, with strict filtering
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                candidate_name = ent.text.strip()
                if 2 <= len(candidate_name.split()) <= 5 and len(candidate_name) > 3:
                    if not (candidate_name.isupper() and len(candidate_name.split()) > 1 and len(candidate_name) > 5):
                        if not any(indicator in candidate_name.lower() for indicator in non_name_indicators):
                            # Prioritize if it appears relatively early in the document
                            if text.find(candidate_name) < 500: # Within first 500 characters
                                name = candidate_name
                                break # Found the best fallback, exit

    return name, email, phone


def extract_skills(text):
    """
    Extracts skills using spaCy's PhraseMatcher and EDLNet predictions,
    with highly refined filtering for precision.
    """
    doc = nlp(text)
    extracted_skills_raw = set() # Use a set to avoid duplicates
    phrase_matcher_skills_lower = set() # Store skills found by PhraseMatcher in lowercase

    text_lower = text.lower() # Pre-calculate for efficiency

    # Define core tech skills that are less prone to false positives from generic contexts
    core_tech_skills_exempt = {
        "python", "java", "sql", "c++", "c#", "javascript", "react", "aws", "azure",
        "git", "docker", "kubernetes", "tensorflow", "pytorch", "hadoop", "spark",
        "linux", "mysql", "postgresql", "mongodb", "oracle", "html", "css", "r",
        "nodejs", "angular", "vue.js", "django", "flask", "spring boot", "rest api",
        "agile", "scrum", "jira", "jenkins", "ansible", "terraform", "selenium", "power bi", "tableau",
        "data mining", "data analysis", # Added these to core_tech_skills_exempt
        "sphinx", "latex", "mathematica", "maple", "cvs", "htcondor" # Added these
    }

    # 1. PhraseMatcher for explicit skill detection and alias resolution
    matches = matcher(doc)
    for match_id, start, end in matches:
        skill_span = doc[start:end]
        skill_text = skill_span.text
        
        # Resolve aliases to canonical form
        canonical_skill = skill_aliases.get(skill_text.lower(), skill_text)
        
        # Add to the set of skills found by PhraseMatcher (in lowercase)
        phrase_matcher_skills_lower.add(canonical_skill.lower())

        # Check context for potential false positives (more aggressive filtering)
        # Expand span for context check
        context_span = doc[max(0, start-20):min(len(doc), end+20)]
        context_text_lower = context_span.text.lower()

        # Negative context indicators (words/phrases that precede non-skill uses)
        # More comprehensive list
        non_skill_contexts = [
            "level of", "basic knowledge of", "understanding of", "exposure to",
            "developed a", "leading a", "in the field of", "area of", "strong in", "expertise in",
            "working with", "used for", "focused on", "responsible for", "demonstrated",
            "familiar with", "experience with", "proficient in", "using", "used", "implementation of",
            "background in", "interest in", "ability to", "concept of", "principle of", "good grasp of"
        ]
        
        is_false_positive_context = False
        
        # If the skill is a single generic word, apply stricter context checks
        if len(canonical_skill.split()) == 1 and canonical_skill.lower() not in core_tech_skills_exempt:
            if any(f"{c} {canonical_skill.lower()}" in context_text_lower for c in non_skill_contexts):
                is_false_positive_context = True
        # For multi-word skills, if they are exactly matched by PhraseMatcher, they are usually good.
        # But if the entire phrase is a generic context (e.g., "experience with python"), apply caution.
        elif len(canonical_skill.split()) > 1:
            # If the skill itself is a common non-skill phrase (e.g., "problem solving" - might be soft skill, not tech)
            if canonical_skill.lower() in ["problem solving", "critical thinking", "communication", "teamwork", "leadership"]:
                # Only include if explicitly listed in a "Skills" section or strong context
                # (This part is harder with just PhraseMatcher, but the overall filtering helps)
                pass # Let post-processing handle this more robustly
            # If a common non-skill context phrase is *immediately* before the skill
            if any(context_phrase in context_text_lower and context_text_lower.find(context_phrase) < context_text_lower.find(canonical_skill.lower()) for context_phrase in non_skill_contexts):
                is_false_positive_context = True

        if not is_false_positive_context:
            extracted_skills_raw.add(canonical_skill)

    # 2. EDLNet predictions (integrate with explicit matches, apply strict filtering)
    edl_predicted_skills_with_confidence, edl_overall_uncertainty = predict_skills_with_edl(text)
    edl_predicted_skills_with_confidence.sort(key=lambda x: x[1], reverse=True) # Sort by confidence

    # Confidence thresholds (tuned even more strictly)
    EDL_VERY_HIGH_CONFIDENCE = 0.96 # Almost certainly a skill
    EDL_HIGH_CONFIDENCE_IN_TEXT = 0.90 # High confidence and appears in text (or matched by phrase matcher)
    EDL_MODERATE_CONFIDENCE_WITH_EXPLICIT_MENTION = 0.80 # Moderate confidence AND explicit mention/phrase match

    # List of broad skills that require strong evidence or very high confidence
    BROAD_SKILLS = ["data science", "artificial intelligence", "machine learning", "big data", "devops", "cybersecurity", "project management", "data mining", "data analysis"] # Added data mining/analysis here
    BROAD_SKILL_INFERENCE_CONFIDENCE = 0.94 # Extremely high confidence needed if inferred
    BROAD_SKILL_EXPLICIT_CONFIDENCE = 0.85 # High confidence if explicitly mentioned

    for skill, confidence in edl_predicted_skills_with_confidence:
        skill_lower = skill.lower()
        canonical_skill = skill_aliases.get(skill_lower, skill) # Use alias mapping for EDL too

        if canonical_skill in extracted_skills_raw: # Already explicitly found, no need to re-add
            continue

        # Rule 1: Very High Confidence - Add unconditionally (EDLNet is almost certain about this inference)
        if confidence >= EDL_VERY_HIGH_CONFIDENCE:
            extracted_skills_raw.add(canonical_skill)
            continue

        # Rule 2: High Confidence AND explicitly present in resume text or found by PhraseMatcher
        if confidence >= EDL_HIGH_CONFIDENCE_IN_TEXT and (skill_lower in text_lower or skill_lower in phrase_matcher_skills_lower):
            extracted_skills_raw.add(canonical_skill)
            continue

        # Rule 3: Specific handling for broad, often inferred skills
        if canonical_skill.lower() in BROAD_SKILLS:
            # Check for explicit mention first (substring or PhraseMatcher hit)
            if canonical_skill.lower() in text_lower or canonical_skill.lower() in phrase_matcher_skills_lower:
                if confidence >= BROAD_SKILL_EXPLICIT_CONFIDENCE:
                    extracted_skills_raw.add(canonical_skill)
            # OR, if it's inferred with extremely high confidence AND strong related evidence (multiple sub-skills)
            elif confidence >= BROAD_SKILL_INFERENCE_CONFIDENCE:
                strong_evidence_found = False
                if canonical_skill.lower() in ["data science", "data mining", "data analysis"]: # Group these
                    if (("pandas" in text_lower or "numpy" in text_lower or "scikit-learn" in text_lower or "tensorflow" in text_lower or "pytorch" in text_lower) and
                        ("statistical analysis" in text_lower or "modeling" in text_lower or "data visualization" in text_lower or "data analysis" in text_lower or "big data" in text_lower or "data mining" in text_lower)) or \
                       ("sql" in text_lower and ("data warehousing" in text_lower or "business intelligence" in text_lower)):
                        strong_evidence_found = True
                elif canonical_skill.lower() == "machine learning":
                    if ("tensorflow" in text_lower or "pytorch" in text_lower or "keras" in text_lower or "scikit-learn" in text_lower) and \
                       ("python" in text_lower or "r" in text_lower) and \
                       ("model training" in text_lower or "algorithm development" in text_lower or "neural networks" in text_lower or "predictive modeling" in text_lower):
                        strong_evidence_found = True
                elif canonical_skill.lower() == "artificial intelligence":
                    if (("machine learning" in text_lower or "deep learning" in text_lower or "nlp" in text_lower or "computer vision" in text_lower) and \
                        ("python" in text_lower or "java" in text_lower or "c++" in text_lower)):
                        strong_evidence_found = True
                elif canonical_skill.lower() == "cybersecurity":
                     if ("siem" in text_lower or "threat detection" in text_lower or "security analytics" in text_lower or \
                         "incident response" in text_lower or "penetration testing" in text_lower or "vulnerability assessment" in text_lower) and \
                        ("network security" in text_lower or "information security" in text_lower or "security operations" in text_lower):
                         strong_evidence_found = True
                elif canonical_skill.lower() == "devops":
                    if ("ci/cd" in text_lower or "docker" in text_lower or "kubernetes" in text_lower or "microservices" in text_lower) and \
                       ("jenkins" in text_lower or "git" in text_lower or "aws" in text_lower or "azure" in text_lower or "terraform" in text_lower or "ansible" in text_lower):
                        strong_evidence_found = True
                elif canonical_skill.lower() == "project management":
                    if ("agile" in text_lower or "scrum" in text_lower or "kanban" in text_lower) and \
                       ("jira" in text_lower or "stakeholder management" in text_lower or "risk management" in text_lower or "budgeting" in text_lower):
                        strong_evidence_found = True

                if strong_evidence_found:
                    extracted_skills_raw.add(canonical_skill)
            continue

        # Rule 4: Moderate Confidence AND explicitly present in resume text (last resort for other skills)
        if confidence >= EDL_MODERATE_CONFIDENCE_WITH_EXPLICIT_MENTION and (skill_lower in text_lower or skill_lower in phrase_matcher_skills_lower):
            extracted_skills_raw.add(canonical_skill)
            continue

    # Final Post-processing: Remove highly generic terms that might have slipped through
    final_skills = set()
    generic_non_skills = {
        "experience", "proficient", "knowledge", "familiarity", "ability",
        "involved", "responsible", "developed", "managed", "worked", "led",
        "implement", "use", "strong", "expert", "good", "excellent",
        "understanding", "overview", "proven", "demonstrated", "skills",
        "frameworks", "tools", "platforms", "technologies", "concepts",
        "approaches", "methodologies", "applications", "research", "analysis",
        "solutions", "system", "processes", "strategy", "innovation", "client",
        "project", "product", "quality", "delivery", "business", "technical",
        "computer", "software", "hardware", "network", "security", "database",
        "development", "design", "operations", "data", "analytics", "engineering",
        "management", "consulting", "support", "customer", "service", "communication",
        "team", "leadership", "problem", "critical", "thinking", "time", "organizational",
        "strategic", "financial", "marketing", "sales", "public", "writing", "documentation",
        "reporting", "compliance", "regulatory", "testing", "assurance", "automation",
        "system", "information", "cloud", "virtualization", "web", "mobile", "user", "interface",
        "role", "key", "advanced", "basic", "intermediate", "expert level"
    }

    # Add soft skills explicitly to the generic list for stricter filtering if not specifically sought
    soft_skills_list = [
        "communication", "teamwork", "leadership", "problem solving", "critical thinking",
        "adaptability", "time management", "negotiation", "presentation skills",
        "stakeholder management", "risk management", "budgeting", "client relations",
        "customer service", "emotional intelligence", "conflict resolution", "decision making",
        "mentoring", "coaching", "report writing", "documentation", "market research",
        "financial modeling", "public speaking", "operations management", "supply chain management",
        "quality management", "change management", "process improvement"
    ]
    generic_non_skills.update(soft_skills_list)


    for skill in extracted_skills_raw:
        skill_lower = skill.lower()
        
        # Remove if it's a single word and a common generic non-skill, AND not a core tech skill
        if len(skill_lower.split()) == 1 and skill_lower in generic_non_skills and skill_lower not in core_tech_skills_exempt:
            continue
        
        # Remove if it's a multi-word phrase but contains only generic non-skills, and is not a canonical broad skill
        # This checks if ALL words in the skill are generic, and the skill itself is not a broad skill like "Project Management"
        if len(skill_lower.split()) > 1 and all(word in generic_non_skills for word in skill_lower.split()) and skill_lower not in BROAD_SKILLS:
            continue

        # Specific check for soft skills: only include if they are explicitly listed as 'skills' or are part of a larger recognized skill phrase
        if skill_lower in soft_skills_list and skill_lower not in core_tech_skills_exempt:
            # Only add if it appeared in a "Skills" section, or if EDLNet was highly confident.
            # This logic needs a "Skills" section detector, which is complex. For now, rely on EDL confidence.
            # If it's a pure soft skill and not highly confident by EDLNet, or not explicitly in 'common_skills' as a *tech* skill, filter it out.
            # This is a heuristic: if a soft skill is extracted, assume EDLnet handles confidence well.
            # For this update, we've added them to generic_non_skills for stricter removal if not strongly detected.
            pass # The generic_non_skills check above should handle most cases.

        final_skills.add(skill)

    return list(final_skills)


def compute_resume_score(resume_text, jd_text):
    """Computes cosine similarity between resume and job description text."""
    if not jd_text:
        return 0.0
    tfidf = TfidfVectorizer(stop_words='english')
    try:
        vectors = tfidf.fit_transform([resume_text, jd_text])
        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return round(score * 100, 2)
    except ValueError:
        return 0.0

def insert_data(name, email, phone, timestamp, skills, score, page_no='1'):
    """Inserts extracted data into the MySQL database."""
    if connection is None or cursor is None:
        st.error("Database connection not established. Cannot save data.")
        return
    try:
        insert_sql = f"""
            INSERT INTO {DB_table_name}
            (Name, Email_ID, Phone, resume_score, Timestamp, Page_no, Skills)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        values = (name, email, phone, str(score), timestamp, page_no, ", ".join(skills))
        cursor.execute(insert_sql, values)
        connection.commit()
    except pymysql.Error as e:
        st.error(f"Error inserting data into database: {e}")

def clear_database():
    """Clears all data from the user_data table."""
    if connection is None or cursor is None:
        st.error("Database connection not established. Cannot clear data.")
        return False
    try:
        # Ask for confirmation before clearing
        if st.session_state.get('confirm_clear_db', False):
            cursor.execute(f"TRUNCATE TABLE {DB_table_name}")
            connection.commit()
            st.success("Database cleared successfully!")
            st.session_state['confirm_clear_db'] = False # Reset confirmation state
            return True
        else:
            # Set a session state variable to indicate confirmation is needed
            st.warning("Are you sure you want to clear ALL data from the database? This action cannot be undone.")
            if st.button("Confirm Clear Database", key="confirm_clear_button"):
                st.session_state['confirm_clear_db'] = True
                st.rerun() # Rerun to trigger the actual clear logic
            return False # Indicate that clearing is pending confirmation
    except pymysql.Error as e:
        st.error(f"Error clearing database: {e}")
        return False

def plot_skill_distribution(skills, title="Skill Distribution (Detected)"):
    """Plots the distribution of detected skills."""
    if not skills:
        st.warning(f"No {title.lower().replace('skill distribution', 'skills')} found to display for distribution.")
        return
    skill_counts = pd.Series(skills).value_counts().head(15) # Limit to top 15 for readability
    fig, ax = plt.subplots(figsize=(10, 6))
    # Fix for FutureWarning: Assign the x variable to hue and set legend=False
    sns.barplot(x=skill_counts.index, y=skill_counts.values, ax=ax, palette='viridis', hue=skill_counts.index, legend=False)
    ax.set_title(title)
    ax.set_xlabel('Skill')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

def admin_dashboard():
    """Displays the Admin Dashboard with database contents and download option."""
    st.title("Admin Dashboard")

    # Initialize session state for authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.subheader("Admin Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")

            if login_button:
                if username == "admin" and password == "admin123":
                    st.session_state.authenticated = True
                    st.success("Logged in successfully!")
                    st.rerun() # Rerun to show the dashboard content
                else:
                    st.error("Invalid Username or Password")
    else:
        # Dashboard content
        st.write("Welcome, Admin!")
        if st.button("Logout", key="logout_button"):
            st.session_state.authenticated = False
            st.info("Logged out successfully.")
            st.rerun()

        st.markdown("---")
        st.subheader("Database Management")
        if st.button("Clear All Data from Database", key="clear_db_button"):
            # This button will now trigger the confirmation flow in clear_database()
            if clear_database(): # If clear_database() returns True (meaning data was cleared)
                st.rerun() # Rerun to refresh the dashboard after clearing

        st.markdown("---")
        st.subheader("Resume Analysis Data")

        if connection is None or cursor is None:
            st.error("Database connection not established. Cannot display data.")
            return

        try:
            cursor.execute(f"SELECT * FROM {DB_table_name}")
            data = cursor.fetchall()
            columns = [i[0] for i in cursor.description]

            if data:
                df = pd.DataFrame(data, columns=columns)
                st.dataframe(df)

                # Option to download data as CSV
                csv_file = df.to_csv(index=False)
                b64 = base64.b64encode(csv_file.encode()).decode()
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                href = f'<a href="data:file/csv;base64,{b64}" download="resume_data_{current_time}.csv">Download Data as CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.info("No data found in the database yet. Upload resumes in the 'Resume Analyzer' section to populate data.")
        except pymysql.Error as e:
            st.error(f"Error fetching data from database: {e}")

def display_pdf(uploaded_file):
    """Displays the uploaded PDF using an embedded iframe."""
    if uploaded_file.type == "application/pdf":
        base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.warning("Cannot display non-PDF files directly here.")

def run_app():
    """Main function to run the Streamlit application."""
    st.set_page_config(layout="wide", page_title="Smart Resume Analyzer")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Resume Analyzer", "Admin Dashboard"])

    if page == "Resume Analyzer":
        st.title("Smart Resume Analyzer with EDL Confidence")
        st.info("Note on Skill Detection: This application uses a *mock* AI model (EDLNet) for skill prediction. While I've made significant improvements, detecting highly contextual skills like 'Data Science' accurately in diverse resumes (e.g., distinguishing 'Python for log analysis' from 'Python for data analysis') is challenging for a keyword-based mock. A real, trained AI model would learn these nuances better.")


        with st.form("resume_form"):
            uploaded_resume = st.file_uploader("Upload Resume", type=["pdf"])
            uploaded_jd = st.file_uploader("Upload Job Description", type=["pdf", "txt"])
            submit = st.form_submit_button("Analyze")

        if submit:
            if not uploaded_resume:
                st.warning("Please upload a resume to analyze.")
                return

            # Store the initial file pointer position
            original_resume_position = uploaded_resume.tell()

            resume_text = extract_text_from_pdf(uploaded_resume)
            uploaded_resume.seek(original_resume_position) # Reset after text extraction

            if not resume_text:
                st.error("Could not extract text from the uploaded resume. Please try a different PDF or ensure it's not an image-only PDF.")
                return

            name, email, phone = extract_info(resume_text)
            skills = extract_skills(resume_text)

            # Check if basic info is detected and provide feedback
            if name == "Unknown" and email == "Unknown" and phone == "Unknown":
                st.warning("Could not detect basic information (Name, Email, Phone) from your resume. This might indicate that the resume is not well-formatted for automated extraction. Please ensure these details are clearly present and in a standard format.")


            page_no = '1'
            try:
                uploaded_resume.seek(original_resume_position) # Ensure pointer is at beginning for pdfplumber
                with pdfplumber.open(uploaded_resume) as pdf:
                    page_no = str(len(pdf.pages))
                uploaded_resume.seek(original_resume_position) # Reset after count
            except Exception as e:
                st.warning(f"Could not determine page count: {e}. Defaulting to 1.")

            jd_text = ""
            if uploaded_jd:
                # Store the initial JD file pointer position
                original_jd_position = uploaded_jd.tell()
                try:
                    if uploaded_jd.name.lower().endswith(".txt"):
                        jd_text = uploaded_jd.read().decode("utf-8")
                    elif uploaded_jd.name.lower().endswith(".pdf"):
                        jd_text = extract_text_from_pdf(uploaded_jd)
                    uploaded_jd.seek(original_jd_position) # Reset JD pointer
                except Exception as e:
                    st.warning(f"Error reading Job Description file: {e}")

            score = compute_resume_score(resume_text, jd_text)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if name != "Unknown" or email != "Unknown" or phone != "Unknown" or skills:
                insert_data(name, email, phone, timestamp, skills, score, page_no)
            else:
                st.warning("Could not extract enough information from the resume to save. Please check the resume format or provide more content.")

            edl_skills_with_confidence, uncertainty = predict_skills_with_edl(resume_text)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Resume Summary")
                st.write(f"**Name**: {name}")
                st.write(f"**Email**: {email}")
                st.write(f"**Phone**: {phone}")
                st.write(f"**Skills Detected**: {', '.join(skills) if skills else 'None'}")
                st.write(f"**Resume Score (Cosine Similarity with JD)**: {score}%")
                st.write(f"**Page Count**: {page_no}")

            with col2:
                st.subheader("EDL Skill Prediction Insights")
                st.write(f"**Overall EDL Uncertainty**: {uncertainty:.2f}")
                st.markdown("---")
                st.markdown("**Top Confident EDL Skills (Inferred)**:")
                if edl_skills_with_confidence:
                    # Sort and display a reasonable number of top skills
                    # Filter out skills with very low confidence for display purposes
                    confident_edl_skills_for_display = [(s, c) for s, c in edl_skills_with_confidence if c >= 0.65]
                    if confident_edl_skills_for_display:
                        confident_edl_skills_for_display.sort(key=lambda x: -x[1])
                        for skill, confidence in confident_edl_skills_for_display[:10]: # Display top 10
                            st.write(f"- {skill}: {confidence:.2f}")
                    else:
                        st.info("No highly confident EDL skills predicted.")
                else:
                    st.warning("No EDL skills predicted.")

            st.subheader("Detected Skill Distribution (from Filtered Extraction)")
            plot_skill_distribution(skills)

            # --- Display Uploaded Resume ---
            st.markdown("---")
            st.subheader("Uploaded Resume Document")
            with st.expander("Click to view uploaded resume"):
                uploaded_resume.seek(original_resume_position) # Important: Reset file pointer before reading for display
                display_pdf(uploaded_resume)
            # --- End Display Uploaded Resume ---


        if st.button("Reset Application"):
            st.experimental_rerun()

    elif page == "Admin Dashboard":
        admin_dashboard()

if __name__ == "__main__":
    if 'confirm_clear_db' not in st.session_state:
        st.session_state['confirm_clear_db'] = False
    run_app()
    if connection:
        try:
            connection.close()
        except pymysql.Error as e:
            st.error(f"Error closing database connection: {e}")
