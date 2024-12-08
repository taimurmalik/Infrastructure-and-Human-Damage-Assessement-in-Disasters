import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from autocorrect import Speller
import emoji
import os

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Initialize spell checker and define stopwords
spell = Speller(lang='en')
stop_words = set(stopwords.words('english'))  # Using a set for efficient lookup

# Define dictionary for decoding abbreviations
abbreviation_map = {
    "thnx": "thanks", "thx": "thanks", "pls": "please", "plz": "please", "btw": "by the way",
    "omg": "oh my god", "idk": "I don't know", "imo": "in my opinion", "brb": "be right back",
    "bff": "best friends forever", "afaik": "as far as I know", "lmk": "let me know",
    "tbh": "to be honest", "np": "no problem", "smh": "shaking my head", "rn": "right now",
    "irl": "in real life", "ftw": "for the win", "fyi": "for your information",
    "gg": "good game", "idc": "I don't care", "nvm": "never mind", "dm": "direct message",
    "msg": "message"
}

# Set file paths
input_path = r"E:/Documents/FYP Data/hurricane_harvey_final_data.xls"
output_path = r"C:/Users/fobai/Desktop/Textualpreprocessing/Hurricane_Harvey.csv"

# Check if the file exists
if not os.path.exists(input_path):
    print(f"File not found at {input_path}")
else:
    # Read file and handle formats
    try:
        if input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        elif input_path.endswith('.xls') or input_path.endswith('.xlsx'):
            df = pd.read_excel(input_path)
        else:
            raise ValueError("Unsupported file format. Please use .csv, .xls, or .xlsx files.")
    except Exception as e:
        print(f"Error reading file: {e}")
        df = pd.DataFrame()  # Initialize empty DataFrame if error occurs

    # Ensure the DataFrame is not empty
    if df.empty:
        print(f"The file at {input_path} is empty.")
    else:
        # Define the text column
        text_column = 'tweet_text'  # Replace with your actual column name

        # Check if the text column exists
        if text_column not in df.columns:
            print(f"Column '{text_column}' not found in the file.")
        else:
            # Text preprocessing function
            def preprocess_text(text):
                if not isinstance(text, str):  # Handle non-string values
                    return ""

                # Remove HTML tags, mentions, hashtags, URLs
                text = re.sub(r'<.*?>|@\w+|#\w+|http\S+|www\S+', '', text)

                # Replace all punctuations with white spaces
                text = re.sub(r'[^\w\s]', ' ', text)

                # Convert to lowercase
                text = text.lower()

                # Remove numbers
                text = re.sub(r'\d+', '', text)

                # Replace emojis with descriptive text
                text = emoji.demojize(text)
                text = re.sub(r':([a-zA-Z_]+):', r'\1', text)

                # Decode abbreviations
                words = text.split()
                text = ' '.join([abbreviation_map.get(word, word) for word in words])

                # Fix misspelled words
                text = ' '.join([spell(word) for word in text.split()])

                # Tokenize text and remove stop words
                text_tokens = word_tokenize(text)
                text = ' '.join([word for word in text_tokens if word not in stop_words])

                # Remove non-ASCII characters
                text = re.sub(r'[^\x00-\x7F]+', '', text)

                return text

            # Apply preprocessing to the tweet_text column
            df['processed_data'] = df[text_column].apply(preprocess_text)

            # Remove duplicate rows based on 'processed_data' column
            df.drop_duplicates(subset=['processed_data'], inplace=True)

            # Drop unnecessary columns
            if 'text_info' in df.columns:
                df.drop('text_info', axis=1, inplace=True)

            # Save the processed data
            df.to_csv(output_path, index=False)
            print(f"Text preprocessing completed and saved to: {output_path}")
