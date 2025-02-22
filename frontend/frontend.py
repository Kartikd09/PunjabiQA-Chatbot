import pandas as pd
import requests
import chainlit as cl

# Step 1: Read Excel File
file_path = "path_to_your_excel_file.xlsx"  # Replace with the actual path to your file
df = pd.read_excel(file_path)

# Ensure the context column exists
if 'context' not in df.columns:
    raise ValueError("The 'context' column is missing in the Excel file.")

# Convert the column to a list
context_data = df['context'].tolist()

# Step 2: Prepare to Send Data to GROQ API
GROQ_API_URL = "https://groq.example.com/api/query"  # Replace with the actual API endpoint
API_KEY = "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"  # Replace with your GROQ API Key

# Step 3: Function to Query GROQ API
def query_groq(question, context_list):
    """
    Query GROQ API with a user question and a dataset context.
    """
    payload = {
        "question": question,
        "context": context_list,  # Provide context list for the query
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(GROQ_API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error querying GROQ API: {response.status_code} {response.text}")

# Step 4: Define State for Theme and Chat History
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
        content="Welcome to the Punjabi  Chatbot! 🖼️\n"
                "Use the button below to toggle between Light and Dark themes."
    ).send()

# Step 5: Handle Theme Toggle Button
@cl.action(label="Toggle Theme")
async def toggle_theme():
    """
    Toggle the theme between light and dark.
    """
    current_theme = cl.user_session.get("theme", "light")
    new_theme = "dark" if current_theme == "light" else "light"
    cl.user_session.set("theme", new_theme)
    await cl.Message(content=f"Theme switched to **{new_theme.capitalize()}** mode!").send()

# Step 6: Handle User Messages and Retain Chat History
@cl.on_message
async def chatbot(user_message):
    """
    Handle user input and maintain chat history.
    """
    try:
        # Retrieve the existing history
        history = cl.user_session.get("history", [])

        # Query the GROQ API
        response = query_groq(user_message.content, context_data)
        answer = response.get('answer', 'No answer found')

        # Add user query and chatbot response to history
        history.append({"user": user_message.content, "bot": answer})
        cl.user_session.set("history", history)

        # Display the response
        await cl.Message(content=answer).send()

        # Optionally, display the full history
        history_message = "\n".join(
            f"**User:** {item['user']}\n**Bot:** {item['bot']}" for item in history
        )
        await cl.Message(content=f"**Chat History:**\n{history_message}").send()

    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()

# Step 7: Launch Chainlit App
# Run `chainlit run chainlit_chatbot.py` to start the chatbot.
