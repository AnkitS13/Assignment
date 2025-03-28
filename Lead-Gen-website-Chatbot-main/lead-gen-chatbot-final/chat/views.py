import os
from langchain_groq import ChatGroq
from langchain.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

from .model import Chat
from airtable import Airtable
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from dotenv import load_dotenv
from .utils import load_web_document, split_docs, scrape
from .prompt import contextualize_q_system_prompt, system_prompt, intent_classification_prompt,lead_prompt_template
import datetime
import re

load_dotenv()

AIRTABLE_API_KEY = os.getenv('AIRTABLE_API_KEY')
AIRTABLE_BASE_KEY = os.getenv('AIRTABLE_BASE_KEY')
AIRTABLE_TABLE_NAME = os.getenv('AIRTABLE_TABLE_NAME')

SMTP_SERVER = os.getenv('SMTP_SERVER')
SMTP_PORT = os.getenv('SMTP_PORT')
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
TO_EMAIL = os.getenv('TO_EMAIL')

airtable = Airtable(
    AIRTABLE_BASE_KEY,
    AIRTABLE_API_KEY
    )

intent_classifier_llm = ChatGroq(model='llama3-8b-8192')

lead_details = {
    "email": "",
    "phone": "",
    "requirements": ""
}

current_question = None
collecting_leads = False
history = []

def send_email(lead_details):
    try:
        
        subject = "New Lead Details"
        body = (
            "You have received new lead details:\n\n"
            f"Email: {lead_details['email']}\n"
            f"Phone: {lead_details['phone']}\n"
            f"Requirements: {lead_details['requirements']}\n"
        )

        message = MIMEMultipart()
        message['From'] = SMTP_USER
        message['To'] = TO_EMAIL
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))

        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, TO_EMAIL, message.as_string())

        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

def save_details(lead_details):
    try:
        airtable.create(AIRTABLE_TABLE_NAME, lead_details)
        print("Data added to Airtable successfully!")
        send_email(lead_details)
    except Exception as e:
        print(f"Error adding data to Airtable or sending email: {e}")

def makeEmb():
    web_pages = scrape()
    docs = load_web_document(web_pages)
    splits = split_docs(docs)
    embedding_model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    vector_store = Chroma.from_documents(
        documents = splits,
        embedding = embedding_model,
        persist_directory = 'chroma_ab',
    )
    return vector_store.as_retriever()

def classify_intent(question):
    global intent_classifier_llm
    chain = intent_classification_prompt | intent_classifier_llm
    response = chain.invoke(question)
    return response.content.strip()


def check_lead_details_in_history():
    global history
    for message in history:
        if isinstance(message, AIMessage) and "We have your details" in message.content:
            return True
    return False

def validate_email(email):
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None

def validate_phone(phone):
    phone_regex = r'^\+?[1-9]\d{1,14}$'  # E.164 format
    return re.match(phone_regex, phone) is not None

def validate_requirements(requirements):
    return len(requirements) > 10

validation_functions = {
    "email": validate_email,
    "phone": validate_phone,
    "requirements": validate_requirements
}

INVALID_RESPONSE_LIMIT = 1
invalid_response_count = 0

def ask_lead_questions(user_response=None):
    global lead_details, current_question, collecting_leads, invalid_response_count

    if check_lead_details_in_history():
        collecting_leads = False
        return "We have your details, we will contact you shortly."

    required_fields = {
        "email": "Email Address",
        "phone": "Phone Number",
        "requirements": "Description of Requirements"
    }

    # Handle user's response
    if user_response:
        for key, label in required_fields.items():
            if lead_details[key] == "" and current_question == label:
                if validation_functions[key](user_response):
                    lead_details[key] = user_response.strip()
                    invalid_response_count = 0  # Reset invalid count on valid input
                else:
                    invalid_response_count += 1
                    if invalid_response_count >= INVALID_RESPONSE_LIMIT:
                        collecting_leads = False
                        invalid_response_count = 0
                        return "I noticed you're having trouble. Let's continue with your other queries for now."
                    return f"The {label} you provided seems invalid. Could you try again?"
                break

    # Find the next missing field
    field_sequence = ["email", "phone", "requirements"]
    for key in field_sequence:
        if not lead_details[key].strip():
            next_field = key
            current_question = required_fields[next_field]
            collecting_leads = True  

            prompt = lead_prompt_template.format(
                user_response=user_response or "",
                lead_details=lead_details,
                missing_field=current_question
            )
            response = intent_classifier_llm.invoke(prompt)
            return response.content.strip()

    # Save details once all fields are collected
    status = save_details(lead_details)
    history.append(AIMessage(content="We have your details, we will contact you shortly."))
    lead_details = {key: "" for key in required_fields}
    current_question = None
    collecting_leads = False  
    return {
        "response": "Thank you for providing your details! Our team will contact you shortly.",
        "status": status
    }

def log_conversation(user_message, bot_response, log_file="chat_logs.txt"):
    try:
        with open(log_file, "a") as file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"[{timestamp}] User: {user_message}\n")
            file.write(f"[{timestamp}] Bot: {bot_response}\n")
            file.write("-" * 50 + "\n")
    except Exception as e:
        print(f"Failed to log conversation: {e}")

def handle_query(query, rag_chain, LLM):
    global collecting_leads, history, invalid_response_count, current_question

    if collecting_leads:
        response = ask_lead_questions(user_response=query)
        log_conversation(query, response)
        if not collecting_leads:
            return handle_query(query, rag_chain, LLM)
        return response

    intent = classify_intent(query)

    if 'company_specific' in intent:
        collecting_leads = False
        current_question = None
        invalid_response_count = 0
        full_response = ""  # Initialize empty response

        for chunk in rag_chain.stream({'input': query, 'chat_history': history}):
            part = chunk.get('answer', '')  # Extract the 'answer' value
            if part:
                print(part, end='', flush=True)  # Print each chunk in real-time
                full_response += part  # Append chunk to the full response

        print()  # Ensure the final print ends with a newline
        history.extend([HumanMessage(content=query), AIMessage(content=full_response)])
        log_conversation(query, full_response)
        return full_response


    elif 'general' in intent:
        collecting_leads = False
        current_question = None
        invalid_response_count = 0
        message = [('system', system_prompt), ('human', query)]
        response = ""
        for chunk in LLM.stream(message):
            response += chunk.content
            print(chunk.content, end='', flush=True)  # Stream to console or UI
        history.extend([HumanMessage(content=query), AIMessage(content=response)])
        log_conversation(query, response)
        return response
    

    elif 'lead_generation' in intent:
        collecting_leads = True
        current_question = None
        invalid_response_count = 0
        response = ask_lead_questions(query)
        log_conversation(query, response)
        return response

    else:
        bot_response = "Sorry, I couldn't understand your query."
        log_conversation(query, bot_response)
        return bot_response


# Define async function for query streaming response
async def stream_response(query: str):
    LLM = ChatGroq(model='llama3-8b-8192')  # Set the model here
    async for chunk in LLM.stream([('human', query)]):
        yield chunk.content


async def chat(_model: Chat):
    try:
        QUERY = _model.query
        
        LLM = ChatGroq(model = 'llama3-8b-8192')

        if os.path.exists('chroma_ab') and any(os.scandir('chroma_ab')):
            embedding_model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
            vector_store = Chroma(
                persist_directory = 'chroma_ab',
                embedding_function = embedding_model
            )
            retriver = vector_store.as_retriever()
        else:
            retriver = makeEmb()
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ('system', contextualize_q_system_prompt),
                MessagesPlaceholder('chat_history'),
                ('human', '{input}'),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            LLM,
            retriver,
            contextualize_q_prompt,
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ('system', system_prompt),
                ('human', '{input}'),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(
            LLM,
            qa_prompt,
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain,
        )

        response = handle_query(QUERY, rag_chain, LLM)

        return {
            'intent': classify_intent(QUERY),
            'response': response
        }

    except Exception as e:
        return {'error': str(e)}