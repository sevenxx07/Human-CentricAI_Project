import os 
import sys
import django
import pandas as pd
import string
import re
import nltk
#nltk.download('wordnet')
#nltk.download('stopwords')
from nltk.corpus import stopwords
#nltk.download('punkt')
#nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 
from textblob import TextBlob

sys.path.append('/Users/stinahellgren/Documents/Human AI/Human-CentricAI_Project')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pbl.settings')
django.setup()

from django.conf import settings

csv_path = os.path.join(settings.BASE_DIR, 'data', 'IMDB Dataset.csv')

print(f"reading data")
df = pd.read_csv(csv_path)

# Lowercasing 

df['review'] = df['review'].str.lower()


# Removing metadata and keeping only text
    # Metadata = data that provides information about other data, but not the content of the data itself

#### TODO ####

# Removing URLs
def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

# Removing HTMLs
def remove_html(text):
    pattern = re.compile(r'<br\s*/?>', re.IGNORECASE)
    return pattern.sub(' ', text)

# Negation handling
def separate_puncuation(text):
    return re.sub(r'([.,!?()"])', r' \1 ', text)

# Acronym and slang expansion <-> acronym and slang list
acronym_slang_dict = {
    "idk": "I do not know",
    "lol": "laughing out loud",
    "brb": "be right back",
    "imo": "in my opinion",
    "btw": "by the way",
    "u": "you",
    "ur": "your",
    "pls": "please",
}

def expand_acronyms(text, acronym_slang_dict):
    tokens = text.split()
    expanded_tokens = []
    for token in tokens: 
        if token in acronym_slang_dict: 
            expanded_tokens.append(acronym_slang_dict[token])
        else:
            expanded_tokens.append(token)
    return ' '.join(expanded_tokens)
    

# Spelling correction <-> spell-checker dictionary
def correct_spelling(text):
    corrected_text = TextBlob(text).correct()
    return str(corrected_text)


def negation_handling(text, scope = 3):
    negation_words = {
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nowhere", "hardly", "scarcely", "barely", "doesn't", 
    "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "won't", "can't", "don't", "didn't", "haven't", "hasn't", "hadn't"
    }
    
    punctuation = {".", ",", ":", ";", "!", "?"}

    tokens = text.split()
    result = []
    negation_counter = 0

    for token in tokens: 
        if token in negation_words:
            negation_counter = scope
            result.append(token)
        elif token in punctuation: 
            negation_counter = 0
            result.append(token)
        elif negation_counter >0:
            result.append(f"{token}_NEG")
            negation_counter -=1
        else: 
            result.append(token)
    return ' '.join(result)
# The words within the scope of the negation get appended with _NEG
# This tells the model not to treat them as their original form

# Removing puncuations 
exclude = set(string.punctuation)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Removing twitter features (hashtags and RT) 
def remove_hashtags(text):
    pattern = re.compile(r'#\S+')
    return pattern.sub(r'', text)

def remove_retweets(text):
    pattern = re.compile(r'^RT @\w+:?\s?', re.IGNORECASE)
    return pattern.sub('', text)

# Removing white spaces from all strings
def remove_white_spaces(text):
    return re.sub(r'\s+',' ', text).strip()

# Anaomyising the text
def anonymize_text(text):
    text = re.sub(r'@\w+', '@user', text) 
    text = re.sub(r'b\\d+\b', '[NUM]', text) 
    text = re.sub(r'\b\S+@\S+\b', '[EMAIL]', text) 
    return text 

# Stop-word removial <-> stop-word list
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return ' '.join([w for w in text if w not in stop_words])

# Short-word removal 
def short_word_removal(text, min_length=3):
    return [word for word in text if len(word)>= min_length]

# Lemmatisation 
wnl = WordNetLemmatizer()

# Clean text ready to be analysed 
def clean_text(text):
    text = remove_url(text)
    text = remove_html(text)
    text = separate_puncuation(text)
    text = expand_acronyms(text, acronym_slang_dict)
    #text = correct_spelling(text)
    text = negation_handling(text)
    text = remove_punctuation(text)
    text = remove_hashtags(text)
    text = remove_retweets(text)
    text = remove_white_spaces(text)
    text = anonymize_text(text)
    text = remove_stopwords(text)
    text = word_tokenize(text) # Tokenisation
    text = short_word_removal(text)
    text = [wnl.lemmatize(word) for word in text]

    return ' '.join(text)
print('Här är vi')
df['review'] = df['review'].apply(clean_text)

print(df['review'][1])
print('hej')
df.to_csv('cleaned_imdb_reviews.csv', index=False)


# Emoticons and emojis tranlation <-> emoticons and emojis dictionary


