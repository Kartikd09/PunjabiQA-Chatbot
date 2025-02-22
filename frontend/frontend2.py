import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chainlit as cl

# Step 1: Load the Dataset
file_path = "path_to_your_excel_file.xlsx"  # Replace with your Excel file path
df = pd.read_excel(file_path)

# Ensure the required columns exist
required_columns = {'questions', 'answers', 'context'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"The Excel file must contain the following columns: {required_columns}")

# Step 2: Vectorize the Questions for Similarity Matching
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df['questions'])

# Step 3: Function to Find the Best Matching Question
def find_best_match(user_query):
    """
    Find the most similar question in the dataset based on cosine similarity.
    """
    user_query_vector = vectorizer.transform([user_query])
    similarities = cosine_similarity(user_query_vector, question_vectors).flatten()
    best_index = np.argmax(similarities)
    return df.iloc[best_index], similarities[best_index]

# Step 4: Query External API with Context
GROQ_API_URL = "https://groq.example.com/api/query"  # Replace with your API URL
API_KEY = "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"  # Replace with your API Key

def query_groq_with_context(question, context):
    """
    Query the external API with the user question and the matched context.
    """
    payload = {
        "question": question,
        "context": context,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    response = requests.post(GROQ_API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get('answer', None)
    else:
        return None

# Step 5: Integrate with Chainlit
@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the chatbot.
    """
    await cl.Message(
        content="Welcome to the Q&A Chatbot! Ask me any question!"
    ).send()

@cl.on_message
async def chatbot(user_message):
    """
    Process the user query and respond.
    """
    user_query = user_message.content

    # Step 5.1: Find the Best Matching Question
    best_match, similarity = find_best_match(user_query)
    matched_question = best_match['questions']
    matched_answer = best_match['answers']
    matched_context = best_match['context']

    # Step 5.2: Query the API Using the Context
    api_answer = query_groq_with_context(user_query, matched_context)

    # Step 5.3: Determine the Final Answer
    final_answer = api_answer if api_answer else matched_answer

    # Step 5.4: Respond with Answer and Info
    response_message = (
        f"**Matched Question:** {matched_question}\n"
        f"**Answer:** {final_answer}"
    )
    await cl.Message(content=response_message).send()
