import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

def download_nltk_module(module_name):
    try:
        nltk.data.find(f'tokenizers/{module_name}')
    except LookupError:
        nltk.download(module_name)

# Load the real data CSV file
real_data = pd.read_csv("DataSet_Misinfo_TRUE.csv", usecols=[1], names=["text"], skiprows=1)
real_data["label"] = "real"

# Load the fake data CSV file
fake_data = pd.read_csv("DataSet_Misinfo_FAKE.csv", usecols=[1], names=["text"], skiprows=1)
fake_data["label"] = "fake"

# Combine the real and fake data into one DataFrame
combined_data = pd.concat([real_data, fake_data], ignore_index=True)
print("Data combined into one DataFrame")

# Download required nltk resources if not already present
download_nltk_module('punkt')
download_nltk_module('stopwords')
download_nltk_module('wordnet')
print("nltk modules downloaded")

# Define preprocessing function
def preprocess_text(text):
    # Check if the input is a string
    if not isinstance(text, str):
        return ""
    
    # Remove newline characters and extra spaces
    text = text.replace('\n', ' ').replace('\r', '').strip()
    
    # Convert text to lowercase
    text = text.lower()

    # Tokenize text
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Perform stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    
    # Reassemble the processed words into a single string
    processed_text = ' '.join(words)
    
    return processed_text

# Apply the preprocessing function to the text column
combined_data['text'] = combined_data['text'].apply(preprocess_text)

print(combined_data.head(5))