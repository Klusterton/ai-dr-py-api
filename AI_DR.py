from fastapi import FastAPI, Form
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import openai

app = FastAPI()

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(
    model='gpt-3.5-turbo'
)

messages = [
    SystemMessage(content="Greetings! I'm your AI Doctor, ready to help. If you're not feeling your best, let's talk about it. Share your symptoms, and we'll work together to figure out the next steps. How can I assist you today?")
]

def print_message(message):
    if isinstance(message, AIMessage):
        print(f"AI Doctor: {message.content}")
    else:
        print(f"{message.role.capitalize()}: {message.content}")

@app.post("/chat")
def chat(user_input: str = Form(...)):
    messages.append(HumanMessage(content=user_input))

    try:
        response = chat(messages)
        print_message(response)
        messages.append(AIMessage(content=response.content))

        if "exit" in response.content.lower():
            return {"message": "Have a nice day. Goodbye!"}

        return {"message": response.content}

    except openai.error.OpenAIError as e:
        if "timeout" in str(e).lower():
            return {"error": "Timeout. The request took too long to process."}
        else:
            raise e
