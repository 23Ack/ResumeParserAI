!pip install nltk
import nltk
nltk.set_proxy('http://proxy.server:port')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.corpus import wordnet
from nltk.metrics import jaccard_distance
import pandas as pd
# Load the dataset (replace 'resumes.csv' with your actual dataset file)
df = pd.read_csv('Resume.csv')

# List of job-related keywords
job_keywords = ['python', 'machine learning', 'data analysis', 'communication', 'problem-solving']
def preprocess_resume(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    return tokens

# Apply preprocessing to each resume in the dataset
df['preprocessed_resume'] = df['resume_text'].apply(preprocess_resume)
def calculate_relevance_score(token_list):
    # Create a frequency distribution of tokens
    fdist = FreqDist(token_list)
    
    # Calculate relevance score based on the number of keyword matches
    score = sum(fdist[key] for key in job_keywords)
    return score

# Apply relevance score calculation to each preprocessed resume
df['relevance_score'] = df['preprocessed_resume'].apply(calculate_relevance_score)
# Sort candidates based on relevance score in descending order
sorted_candidates = df.sort_values(by='relevance_score', ascending=False)

# Display the top candidates
top_candidates = sorted_candidates.head(5)  # You can change the number of candidates to display
top_candidates[['applicant_name', 'relevance_score']]
