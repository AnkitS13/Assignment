import os
from langchain_groq import ChatGroq
from langchain.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
# from sentence_transformers import SentenceTransformer
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
    "name": "",
    "email": "",
    "company": "",
    "requirements": ""
}

lead_details = {"name": "", "email": "", "company": "", "requirements": ""}
current_question = None
collecting_leads = False  # Flag to track if we are collecting lead details
history = []  # History of interactions

def send_email(lead_details):
    try:
        # Create the email message
        subject = "New Lead Details"
        body = (
            "You have received new lead details:\n\n"
            f"Name: {lead_details['name']}\n"
            f"Email: {lead_details['email']}\n"
            f"Company: {lead_details['company']}\n"
            f"Requirements: {lead_details['requirements']}\n"
        )

        message = MIMEMultipart()
        message['From'] = SMTP_USER
        message['To'] = TO_EMAIL
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))

        # Send the email
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
    """
    Checks if the lead details are already provided in the history.
    Returns True if details are found, otherwise False.
    """
    global history
    for message in history:
        if isinstance(message, AIMessage) and "We have your details" in message.content:
            return True
    return False


def ask_lead_questions(user_response=None):
    global lead_details, current_question, collecting_leads

    # Check if lead details are already in history
    if check_lead_details_in_history():
        collecting_leads = False
        return "We have your details, we will contact you shortly."

    # Define the required fields in order
    required_fields = {
        "name": "Full Name",
        "email": "Email Address",
        "company": "Company Name",
        "requirements": "Description of Requirements"
    }

    # Update the current question's response if provided
    if user_response:
        for key, label in required_fields.items():
            if lead_details[key] == "" and current_question == label:
                lead_details[key] = user_response.strip()
                break

    # Find the first missing field in the predefined order
    field_sequence = ["name", "email", "company", "requirements"]
    for key in field_sequence:
        if not lead_details[key].strip():
            next_field = key
            current_question = required_fields[next_field]
            collecting_leads = True  # Enable lead collection phase

            # Format the prompt with the next missing field
            prompt = lead_prompt_template.format(
                user_response=user_response or "",
                lead_details=lead_details,
                missing_field=current_question
            )
            response = intent_classifier_llm.invoke(prompt)
            return response.content.strip()

    # All fields are filled; reset the state and add confirmation to history
    status = save_details(lead_details)
    history.append(AIMessage(content="We have your details, we will contact you shortly."))
    lead_details = {key: "" for key in required_fields}
    current_question = None
    collecting_leads = False  # End lead collection phase
    return {
        "response": "Thank you for providing your details! Our team will contact you shortly.",
        "status": status
    }



def handle_query(query, rag_chain, LLM):
    global collecting_leads, history

    if collecting_leads:
        return ask_lead_questions(user_response=query)

    intent = classify_intent(query)

    if 'company_specific' in intent:
        response = rag_chain.invoke({
            'input': query,
            'chat_history': history
        })
        history.extend([
            HumanMessage(content = query),
            AIMessage(content = response.get('answer', '')),
        ])
        return response.get('answer', 'Sorry, I couldn\'t find an answer.')
    
    elif 'general' in intent:
        response = LLM.invoke(query)
        history.extend([
            HumanMessage(content = query),
            AIMessage(content = response.content.strip()),
        ])
        return response.content.strip()
    
    elif 'lead_generation' in intent:
        collecting_leads = True
        return ask_lead_questions(query)
    
    else:
        return 'Sorry, I couldn\'t understand your query.'


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