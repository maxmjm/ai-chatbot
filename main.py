from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Template for conversation with placeholders 
template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""

# Initialise LLM model and create a ChatPromptTemplate
model = OllamaLLM(model="llama3") 
prompt = ChatPromptTemplate.from_template(template)

# Combine prompt and model into a chain for interaction
chain = prompt | model

# Function to handle the chatbot's conversation loop
def handle_conversation():
    context = ""  # Empty context to store conversation history
    print("Welcome to the AI Chatbot, type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        try:
            # Invoke chain with current context and user question
            result = chain.invoke({"context": context, "question": user_input})

            print("Bot:", result.strip())

            # Append current interaction to conversation history
            context += f"\n User: {user_input}\nAI: {result.strip()}"

        except Exception as e:
            print("Bot: Sorry, I encountered an error:", str(e))

# Entry point of script
if __name__ == "__main__":
    handle_conversation()
