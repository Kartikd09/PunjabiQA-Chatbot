import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chainlit as cl

# Step 1: Read the new Excel dataset
file_path = "./qa-pairs - Copy - empty removed.xlsx"  # Replace with the path to your new dataset
df = pd.read_excel(file_path)

# Ensure the dataset has the necessary columns
if not all(col in df.columns for col in ['question', 'answer', 'context']):
    raise ValueError("The dataset must contain 'question', 'answer', and 'context' columns.")

# Step 2: Preprocess the dataset
question = df['question'].tolist()
answer = df['answer'].tolist()
contexts = df['context'].tolist()

# Step 3: Function to find the most similar question from the dataset
def find_best_matching_question(user_question, question):
    """
    Find the best matching question using cosine similarity.
    """
    vectorizer = TfidfVectorizer().fit_transform([user_question] + question)  # Include the user's question
    similarity_matrix = cosine_similarity(vectorizer[0:1], vectorizer[1:])  # Compare with all question
    best_match_idx = similarity_matrix.argmax()  # Get the index of the best match
    return best_match_idx, question[best_match_idx], contexts[best_match_idx]

# Step 4: Function to query Llama-3.1-70b-versatile model on GROQ API using the matched context
GROQ_API_URL = "https://groq.example.com/api/v1/query"  # Replace with your actual GROQ API URL for Llama-3.1-70b-versatile
API_KEY = "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"  # Replace with your actual GROQ API Key

def query_llama_model(user_question, context):
    """
    Use GROQ's Llama-3.1-70b-versatile model to get an answer based on the user question and context.
    """
    payload = {
        "model": "llama-3.1-70b-versatile",  # Specify the Llama model in the request payload
        "question": user_question,
        "context": context,  # The context related to the best matching question
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Make a request to the GROQ API to query the model
    response = requests.post(GROQ_API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()  # Assuming the answer is returned as part of the response
    else:
        raise Exception(f"Error querying GROQ API: {response.status_code} {response.text}")

# Step 5: Define State for Theme and Chat History
@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the chatbot state with default values.
    """
    # Set the initial theme to 'light'
    cl.user_session.set("theme", "light")
    # Initialize an empty chat history
    cl.user_session.set("history", [])
    # Display welcome message
    await cl.Message(
        content="Welcome to the Punjabi Artists Chatbot! 🖼️\n"
                "You can ask me question about Punjabi artists."
    ).send()

# Step 6: Handle User Messages and Answer Based on Best Match
@cl.on_message
async def chatbot(user_message):
    """
    Handle user input, find the most relevant question, and respond based on context.
    """
    try:
        # Find the best matching question from the dataset
        best_match_idx, matched_question, matched_context = find_best_matching_question(
            user_message.content, question
        )

        # Query the Llama-3.1-70b-versatile model with the matched context to get an answer
        response = query_llama_model(user_message.content, matched_context)
        
        # Extract the answer from the response (this depends on the actual response format)
        answer = response.get('answer', 'No answer found')

        # Store the query and the corresponding response
        history = cl.user_session.get("history", [])
        history.append({"user": user_message.content, "bot": answer})
        cl.user_session.set("history", history)

        # Send the answer back to the user
        await cl.Message(content=answer).send()

        # Optionally, display the chat history
        history_message = "\n".join(
            f"**User:** {item['user']}\n**Bot:** {item['bot']}" for item in history
        )
        await cl.Message(content=f"**Chat History:**\n{history_message}").send()

    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()

# Step 7: Launch Chainlit App
# Run `chainlit run chainlit_chatbot.py` to start the chatbot.
