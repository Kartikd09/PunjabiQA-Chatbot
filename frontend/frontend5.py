import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chainlit as cl
from groq import Groq
import os
from dotenv import load_dotenv


file_path = "../cleaned_qa_pairs.xlsx" 
df = pd.read_excel(file_path)


if not all(col in df.columns for col in ['question', 'answer', 'context']):
    raise ValueError("The dataset must contain 'question', 'answer', and 'context' columns.")


question = df['question'].tolist()
answer = df['answer'].tolist()
contexts = df['context'].tolist()


def find_best_matching_question(user_question, question):
    """
    Find the best matching question using cosine similarity.
    """
    vectorizer = TfidfVectorizer().fit_transform([user_question] + question)  
    similarity_matrix = cosine_similarity(vectorizer[0:1], vectorizer[1:])  
    best_match_idx = similarity_matrix.argmax()  
    return best_match_idx, question[best_match_idx], contexts[best_match_idx], similarity_matrix[0][best_match_idx]


load_dotenv()  
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
client = Groq(api_key=GROQ_API_KEY)

def query_llama_model_with_groq(user_question, context=None):
    """
    Use the GROQ Llama-3.1-70b-versatile model to get an answer based on the user question and context.
    """
    input_prompt = f"""Answer the 
    Question: {user_question}
    """
    if context:
        input_prompt = f"""
        Context: {context}
        {input_prompt}
        """
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": input_prompt}
        ],
        model="llama-3.1-70b-versatile"  
    )
    return chat_completion.choices[0].message.content

@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the chatbot state with default values.
    """
    
    cl.user_session.set("theme", "light")
    cl.user_session.set("history", [])
    
    await cl.Message(
        content="Welcome to the Punjabi  Chatbot! \n"
    ).send()


@cl.on_message
async def chatbot(user_message):
    """
    Handle user input, find the most relevant question, and respond based on context.
    """
    try:
        
        best_match_idx, matched_question, matched_context, similarity = find_best_matching_question(
            user_message.content, question
        )

        if similarity < 0.5:
            
            response = query_llama_model_with_groq(user_message.content)
        else:
            
            response = query_llama_model_with_groq(user_message.content, matched_context)
        
        history = cl.user_session.get("history", [])
        history.append({"user": user_message.content, "bot": response})
        cl.user_session.set("history", history)

        await cl.Message(content=response).send()

    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()

