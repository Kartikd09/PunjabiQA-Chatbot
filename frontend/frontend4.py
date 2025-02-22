import pandas as pd
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
from indicnlp.tokenize import sentence_tokenize
import re

# Step 1: Read the new Excel dataset
file_path = "./cleaned_qa_pairs.xlsx"  # Replace with the path to your new dataset
df = pd.read_excel(file_path)

# Ensure the dataset has the necessary columns
if not all(col in df.columns for col in ['question', 'answer', 'context']):
    raise ValueError("The dataset must contain 'question', 'answer', and 'context' columns.")

# Step 2: Preprocess the dataset
question = df['question'].tolist()
answer = df['answer'].tolist()
contexts = df['context'].tolist()

# Preprocess for both languages
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.strip())  # Remove punctuation and extra spaces
    return text

# Function to detect language (Punjabi or English)
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'  # Default to English if detection fails

# Preprocess the question based on detected language
def preprocess_for_language(text):
    language = detect_language(text)
    if language == 'pa':  # Punjabi
        return preprocess_text(text)
    else:  # English
        return preprocess_text(text)

# Tokenize for both languages
def tokenize_for_language(text, language='en'):
    if language == 'pa':  # Punjabi
        return sentence_tokenize(text)
    else:  # English
        return text.split()  # Default word tokenization for English

# Function to find the most similar question from the dataset
def find_best_matching_question(user_question, question, language='en'):
    """
    Find the best matching question using cosine similarity.
    """
    # Preprocess and tokenize based on language
    user_question = preprocess_for_language(user_question)
    question = [preprocess_for_language(q) for q in question]
    
    vectorizer = TfidfVectorizer(tokenizer=lambda x: tokenize_for_language(x, language)).fit_transform([user_question] + question)
    similarity_matrix = cosine_similarity(vectorizer[0:1], vectorizer[1:])
    best_match_idx = similarity_matrix.argmax()
    return best_match_idx, question[best_match_idx], contexts[best_match_idx], similarity_matrix[0][best_match_idx]

# Query the model (e.g., Llama) to get an answer in the appropriate language
def query_llama_model_with_groq(user_question, context=None, language='en'):
    """
    Query the model based on the language and context.
    """
    input_prompt = f"Answer in {'Punjabi' if language == 'pa' else 'English'}:\nQuestion: {user_question}\n"
    if context:
        input_prompt += f"Context: {context}\n"

    # Simulating Llama model response (you would integrate the actual model call here)
    response = "This is a placeholder response from the model."

    if language == 'pa':
        response = translate_to_punjabi(response)  # Translate response to Punjabi if necessary
    return response

# Translate response to Punjabi if the model outputs in English
def translate_to_punjabi(text):
    """
    Translate English text to Punjabi using DeepL Translator.
    """
    translator = GoogleTranslator(source='en', target='pa')
    return translator.translate(text)

# Handling user messages and answering based on best match
def chatbot(user_message):
    """
    Handle user input, find the most relevant question, and respond based on context.
    """
    try:
        # Detect language of the input
        language = detect_language(user_message)
        
        # Find the best matching question from the dataset
        best_match_idx, matched_question, matched_context, similarity = find_best_matching_question(user_message, question, language)

        # If similarity is below a threshold (e.g., 0.5), treat it as an unmatched question
        if similarity < 0.5:
            # Query the model directly without context if there's no good match
            response = query_llama_model_with_groq(user_message, language=language)
        else:
            # Query the model with the matched context to get an answer
            response = query_llama_model_with_groq(user_message, matched_context, language=language)

        # Send the answer back to the user
        return response

    except Exception as e:
        return f"Error: {str(e)}"

# Example user inputs
user_message1 = "ਭਗਤ ਸਿੰਘ ਕੌਣ ਸੀ?"  # Example Punjabi query
user_message2 = "Who was Bhagat Singh?"  # Example English query

# Testing the chatbot
response1 = chatbot(user_message1)
response2 = chatbot(user_message2)

print(f"Response to Punjabi query: {response1}")
print(f"Response to English query: {response2}")
