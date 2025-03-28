import os
from utils import store_docs, get_response, get_chroma_client

# Store the documents
store_docs("https://nebula9.ai/")

# Setup environment variable for Groq API key
os.environ["GROQ_API_KEY"] = "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"

vector_store = get_chroma_client()
lis = vector_store.get(include=['embeddings','metadatas','documents'])

# Define organization information
organization_name = "Nebula9.ai"
organization_info = (
    "We offer a diverse range of services that form the backbone of our AI solutions. "
    "From Artificial Intelligence & Machine Learning to Quality Assurance and Cloud Services, "
    "we provide end-to-end solutions designed for your success."
)
contact_info = """Email: info@nebula9.ai
India Phone: +91 9999032126
International Phone: +1 (412) 568-3901.
Book a consultation: https://nebula9.ai/book-a-free-consultation/
Get In Touch: https://nebula9.ai/contact-us//"""

# Terminal-based chatbot loop
def chatbot_terminal():
    print("Welcome to the Nebula9.ai Chatbot!")
    print("You can ask questions about the company. Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        response = get_response(user_input, organization_name, organization_info, contact_info)
        print(f"Chatbot: {response}\n")

# Run the chatbot in the terminal
if __name__ == "__main__":
    chatbot_terminal()
