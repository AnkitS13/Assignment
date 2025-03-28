import os
import bs4
import json
import re
import csv
from scrape_url import web_pages
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage


os.environ["GROQ_API_KEY"] = "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"


llm = ChatGroq(model="llama3-8b-8192")

# Load documents from the web pages
loader = WebBaseLoader(web_paths=web_pages)
docs = loader.load()

# Split documents into chunks for better processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = []

for doc in docs:
    
    doc_chunks = text_splitter.split_documents([doc])
    
    
    for idx, chunk in enumerate(doc_chunks):
        chunk.metadata["chunk_index"] = idx
        splits.append(chunk)


def clean_text(text):
    
    text = text.replace("\t", " ").replace("\n", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text


scraped_data = []


for doc in splits:
    cleaned_text = clean_text(doc.page_content)
    scraped_data.append({
        "page_url": doc.metadata.get("source", "Unknown"), 
        "title": doc.metadata.get("title", "No Title Available"),
        "text": cleaned_text,                         
        "chunk_index": doc.metadata.get("chunk_index", 0),
        "word_count": len(cleaned_text.split())
    })

# # Save the data to a JSON file
# output_file = "langchain_scraped_data.json"
# with open(output_file, "w", encoding="utf-8") as json_file:
#     json.dump(scraped_data, json_file, ensure_ascii=False, indent=4)

# print(f"Scraped data saved to {output_file}.")


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding_model,
    persist_directory="chroma_db"  
)


retriever = vectorstore.as_retriever()

# Persist the database to reuse it later
#vectorstore.persist()

# Contextualize questions with history-aware retriever
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


system_prompt = (
    "You are an expert support agent at Nebula9.ai, which offers a diverse range of services that form the backbone of our AI solutions. "
    "From Artificial Intelligence & Machine Learning to Quality Assurance and Cloud Services, "
    "we provide end-to-end solutions designed for your success. "
    "Your task is to answer customer queries related to Nebula9.ai. You should always talk positively about Nebula9.ai and emphasize that it is the best in the industry. "
    "You should also highlight that the customer is making the best decision by purchasing our products. "
    "You should never mention any other company, website, resource, book, tool, or product not related to Nebula9.ai. "
    "Always promote Nebula9.ai’s products. If you don't know the answer, don't try to make up an answer. Just say that you don't know and advise the customer to contact company support. "
    "The ways to contact company support are: Email: info@nebula9.ai, India Phone: +91 9999032126, International Phone: +1 (412) 568-3901, "
    "Book a consultation: https://nebula9.ai/book-a-free-consultation/, Get In Touch: https://nebula9.ai/contact-us/. "
    "The head of Tech Engineering and Innovation is Kamal Chawla and head of AI delivery is Mudit Sharma."
    "Don't be overconfident and avoid hallucinating. Ask follow-up questions if necessary, or if there are several offerings related to the user's query. "
    "Provide answers with complete details in a properly formatted manner with working links and resources wherever applicable within the company's website. "
    "Never provide wrong links.\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# intent_classification_prompt = ChatPromptTemplate.from_messages([
#     ("system", (
#         "You are an intent classifier for user queries. Your task is to classify each question as one of the following intents:\n\n"
#         "1. 'company_specific': The question is directly related to Nebula9.ai's services, products, or offerings. These include topics such as:\n"
#         "   - Services provided by Nebula9.ai (e.g., AI, Cloud, Consulting, etc.)\n"
#         "   - Details about the company’s contact methods, engagement models, or consulting approaches.\n"
#         "   - Case studies, blogs, or specific content hosted on Nebula9.ai's website.\n"
#         "   - Information about careers, the company, or any other content specific to Nebula9.ai.\n"
#         "   - Any other query that requires using the RAG pipeline for answering with verified company-related information.\n\n"
#         "2. 'general': The question is general and not specifically related to Nebula9.ai. Examples include:\n"
#         "   - Questions about generic AI concepts (e.g., 'What is machine learning?')\n"
#         "   - Industry trends or news not tied to Nebula9.ai (e.g., 'What are the latest trends in cloud computing?')\n"
#         "   - Hypothetical questions or unrelated topics (e.g., 'What is the future of AI in healthcare?').\n\n"
#         "   - Any salutation/greetings/ general messages which a normal chatbot should answer.\n\n"
#         "Classify the question into one of the two categories: 'company_specific' or 'general'. Do not provide an answer to the question itself. Simply classify it."
#     )),
#     ("human", "{input}")
# ])

intent_classification_prompt = ChatPromptTemplate.from_template(
    """
    You are an intent classifier for user queries. Your task is to classify each question as one of the following intents:\n\n
    1. 'company_specific': The question is directly related to Nebula9.ai's services, products, or offerings. These include topics such as:\n
       - Services provided by Nebula9.ai (e.g., AI, Cloud, Consulting, etc.)\n
       - Details about the company’s contact methods, engagement models, or consulting approaches.\n
       - Case studies, blogs, or specific content hosted on Nebula9.ai's website.\n
       - Information about careers, the company, or any other content specific to Nebula9.ai.\n
       - Any other query that requires using the RAG pipeline for answering with verified company-related information.\n\n
    2. 'general': The question is general and not specifically related to Nebula9.ai. Examples include:\n
       - Questions about generic AI concepts (e.g., 'What is machine learning?').\n
       - Industry trends or news not tied to Nebula9.ai (e.g., 'What are the latest trends in cloud computing?').\n
       - Hypothetical questions or unrelated topics (e.g., 'What is the future of AI in healthcare?').\n\n
       - Any salutation/greetings/general messages that a typical chatbot should answer.\n\n
    3. 'lead_generation': The query or message indicates the user is interested in Nebula9.ai's services and requires follow-up action. Examples include:\n
       - Inquiries about pricing, collaboration, partnerships, or consulting services (e.g., 'How much does it cost to use your services?').\n
       - Questions that indicate intent to purchase or engage (e.g., 'How do I start using your AI solutions?' or 'Can I schedule a demo?').\n
       - Statements of interest (e.g., 'I want to learn more about your offerings,' 'We need AI solutions for our company,' or 'I am looking for consulting in cloud services.').\n
       - Explicit requests to connect or follow up (e.g., 'Can someone from your team contact me?' or 'Let’s discuss partnership opportunities.').\n
       - Questions related to services (e.g., 'I’m interested in consulting for cloud services.', 'Can I schedule a demo for your AI solutions?').\n
       - Requests for demos, trials, or detailed service descriptions (e.g., 'Can I schedule a demo for your cloud solutions?').\n
       - Any query expressing the need for sales-related information or a request to engage further with Nebula9.ai's team.\n\n
    Classify the question into one of these three categories: 'company_specific', 'general', or 'lead_generation'. 
    Do not provide an answer to the question itself. Simply classify it.\n\n
    Question: {input}
    """
)


intent_classifier_llm = ChatGroq(model="llama3-8b-8192")

def classify_intent(question):
    """Classifies the intent of the question."""
    chain = intent_classification_prompt | llm
    response = chain.invoke(question)
    # response = intent_classifier_llm.invoke({"input": question}, intent_classification_prompt)
    return response.content.strip()


lead_details = {}

def ask_lead_questions():
    """Collect lead information in one prompt and ensure all required details are provided."""
    required_fields = ["name", "email", "company", "requirements"]
    details = {}

    while True:
        user_input = input(
            "Could you please provide the following details?\n"
            "Name, Email, Company, and Requirements (e.g., 'John Doe, john.doe@example.com, ABC Corp, AI solutions'):\n"
        ).strip()

    
        user_response = [item.strip() for item in user_input.split(",")]
        
        # Check if the number of responses matches the required fields
        if len(user_response) == len(required_fields):
            details = dict(zip(required_fields, user_response))
            break
        else:
            print(
                "It seems some information is missing. Please provide all details in the format:\n"
                "'Name, Email, Company, Requirements'\n"
            )

    # Store the lead details in a CSV file
    lead_details[details["email"]] = details

    file_name = "leads.csv"
    file_exists = os.path.isfile(file_name)
    
    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=required_fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(details)

    response = (
        f"Thank you, {details['name']}! "
        "We appreciate your interest in Nebula9.ai. "
        "Our team will reach out to you shortly at the provided email. If you have any further questions, feel free to ask!"
    )
    return response


def handle_query(question, chat_history):
    """Handles the user query based on the intent classification."""
    intent = classify_intent(question)
    
    if 'company_specific' in intent:
        response = rag_chain.invoke({"input": question, "chat_history": chat_history})
        chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=response.get("answer", "")),  
        ])
        return response.get("answer", "Sorry, I couldn't find an answer.")

    elif 'general' in intent:
        response = llm.invoke(question)
        chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=response.content),  
        ])
        return response.content

    elif 'lead_generation' in intent:
        response = rag_chain.invoke({"input": question, "chat_history": chat_history})
        chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=response.get("answer", "")),  
        ])
        lead_questions_response = ask_lead_questions()
        combined_response = f"{response.get('answer', '')}\n\n{lead_questions_response}"

        
        chat_history.append(AIMessage(content=lead_questions_response))

        return combined_response  

    return "Sorry, I couldn't understand your question."


chat_history = []

# company-specific
# question_1 = "What are the services offered by Nebula9.ai?"
# response_1 = handle_query(question_1, chat_history)
# print("Q1:", question_1)
# print("A1:", response_1)

# # general
# question_2 = "What is artificial intelligence?"
# response_2 = handle_query(question_2, chat_history)
# print("Q2:", question_2)
# print("A2:", response_2)

# # Lead generation query
# question_3 = "I am interested in your services."
# response_3 = handle_query(question_3, chat_history)
# print("Q3:", question_3)
# print("A3:", response_3)

# print("\nCollected Leads:", lead_details)

def chatbot_interface():
    """Terminal-based chatbot interface."""
    print("Welcome to Nebula9.ai Chatbot!")
    print("Feel free to ask any questions about our services or offerings.")
    print("Type 'exit' to end the chat.\n")

    chat_history = []

    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == "exit":
            print("Chatbot: Thank you for chatting with Nebula9.ai! Have a great day!")
            break
        try:
            response = handle_query(user_input, chat_history)
            print(f"Chatbot: {response}\n")
        except Exception as e:
            print(f"Chatbot: An error occurred: {e}\n")

chatbot_interface()
