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
import streamlit as st

os.environ["GROQ_API_KEY"] = "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"

if 'llm' not in st.session_state:
    st.session_state.llm = ChatGroq(model="llama3-8b-8192")

# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# if 'docs' not in st.session_state:
#     loader = WebBaseLoader(web_paths=web_pages)
#     st.session_state.docs = loader.load()

# if 'splits' not in st.session_state:
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     splits = []
#     for doc in st.session_state.docs:
#         doc_chunks = text_splitter.split_documents([doc])
#         for idx, chunk in enumerate(doc_chunks):
#             chunk.metadata["chunk_index"] = idx
#             splits.append(chunk)
#     st.session_state.splits = splits

# def clean_text(text):
#     text = text.replace("\t", " ").replace("\n", " ")
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# if 'scraped_data' not in st.session_state:
#     scraped_data = []
#     for doc in st.session_state.splits:
#         cleaned_text = clean_text(doc.page_content)
#         scraped_data.append({
#             "page_url": doc.metadata.get("source", "Unknown"), 
#             "title": doc.metadata.get("title", "No Title Available"),
#             "text": cleaned_text,                         
#             "chunk_index": doc.metadata.get("chunk_index", 0),
#             "word_count": len(cleaned_text.split())
#         })
#     st.session_state.scraped_data = scraped_data

persist_directory = "chroma_db"

# Initialize vectorstore and retriever
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

# Load or create the vectorstore
if st.session_state.vectorstore is None:
    if os.path.exists(persist_directory) and any(os.scandir(persist_directory)):
        # Load the existing vectorstore
        st.info("Loading existing vectorstore from the directory...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vectorstore = Chroma(
            persist_directory=persist_directory, 
            embedding_function=embedding_model
        )
    else:
        # Check if splits are available to create a new vectorstore
        if 'splits' in st.session_state and st.session_state.splits:
            st.info("Creating a new vectorstore...")
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vectorstore = Chroma.from_documents(
                documents=st.session_state.splits,
                embedding=embedding_model,
                persist_directory=persist_directory
            )
        else:
            st.error("No data available to create or load a vectorstore.")

# Initialize the retriever if not already set
if st.session_state.vectorstore and st.session_state.retriever is None:
    st.session_state.retriever = st.session_state.vectorstore.as_retriever()

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

if 'contextualize_q_prompt' not in st.session_state:
    st.session_state.contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

if 'history_aware_retriever' not in st.session_state:
    st.session_state.history_aware_retriever = create_history_aware_retriever(
        st.session_state.llm, st.session_state.retriever, st.session_state.contextualize_q_prompt
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

if 'qa_prompt' not in st.session_state:
    st.session_state.qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

if 'question_answer_chain' not in st.session_state:
    st.session_state.question_answer_chain = create_stuff_documents_chain(st.session_state.llm, st.session_state.qa_prompt)
    st.session_state.rag_chain = create_retrieval_chain(st.session_state.history_aware_retriever, st.session_state.question_answer_chain)

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
       - Explicit requests to connect or follow up (e.g., 'Can someone from your team contact me?' or 'Let’s discuss partnership opportunities.').\n\n
       - Questions related to services (e.g., 'I’m interested in consulting for cloud services.', 'Can I schedule a demo for your AI solutions?').\n\n
       - Requests for demos, trials, or detailed service descriptions (e.g., 'Can I schedule a demo for your cloud solutions?').\n\n
       - Any query expressing the need for sales-related information or a request to engage further with Nebula9.ai's team.\n\n
    Classify the question into one of these three categories: 'company_specific', 'general', or 'lead_generation'. 
    Do not provide an answer to the question itself. Simply classify it.\n\n
    Question: {input}
    """
)

if 'intent_classifier_llm' not in st.session_state:
    st.session_state.intent_classifier_llm = ChatGroq(model="llama3-8b-8192")

def classify_intent(question):
    chain = intent_classification_prompt | st.session_state.intent_classifier_llm
    response = chain.invoke(question)
    return response.content.strip()

lead_prompt_template = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant guiding the user through a tailored inquiry process. Your goal is to gather the following information naturally and conversationally:
    - Full Name
    - Email Address
    - Company Name
    - Description of their requirements or needs

    Engage in friendly, personalized dialogue to collect the required information. Adjust your responses to ensure the user feels heard and valued:
    - Respond appropriately to their answers, addressing any concerns or questions.
    - If any information is missing, gently and seamlessly guide the conversation to gather it.

    Once all the details are provided, confirm politely that you’ve noted everything and express appreciation for their time and input.
    ---
    User Response: {user_response}
    Current Details Collected: {lead_details}
    Missing Information: {missing_field}
    """
)


def ask_lead_questions(user_response=None):
    required_fields = {
        "name": "Full Name",
        "email": "Email Address",
        "company": "Company Name",
        "requirements": "Description of Requirements"
    }

    if "lead_details" not in st.session_state:
        st.session_state.lead_details = {key: "" for key in required_fields}

    # Update the current question's response if provided
    if user_response:
        for key, field in required_fields.items():
            if not st.session_state.lead_details[key]:
                st.session_state.lead_details[key] = user_response
                break

    # Check for missing fields
    missing_fields = [key for key, value in st.session_state.lead_details.items() if not value]
    if missing_fields:
        next_field = missing_fields[0]
        # Use the lead-specific prompt template
        prompt = lead_prompt_template.format(
            user_response=user_response or "",
            lead_details=st.session_state.lead_details,
            missing_field=required_fields[next_field]
        )
        response = st.session_state.intent_classifier_llm.invoke(prompt)
        st.session_state.current_question = required_fields[next_field]
        return response.content.strip()

    # All details collected
    file_name = "leads.csv"
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=st.session_state.lead_details.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(st.session_state.lead_details)

    # Reset the state and return a success message
    st.session_state.lead_details = {key: "" for key in required_fields}
    st.session_state.current_question = ""
    return "Thank you for providing your details! Our team will contact you shortly."


def handle_query(question):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    

     # Handle responses to current lead questions
    if "current_question" in st.session_state and st.session_state.current_question:
        response = ask_lead_questions(user_response=question)
        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=response))
        return response
    
    intent = classify_intent(question)

    if 'company_specific' in intent:
        response = st.session_state.rag_chain.invoke({"input": question, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=response.get("answer", "")),
        ])
        return response.get("answer", "Sorry, I couldn't find an answer.")
    
    elif 'general' in intent:
        response = st.session_state.llm.invoke(question)
        st.session_state.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=response.content),
        ])
        return response.content
    
    elif 'lead_generation' in intent:
        st.session_state.chat_history.append(HumanMessage(content=question))
        response = ask_lead_questions()
        st.session_state.chat_history.append(AIMessage(content=response))
        return response

    return "Sorry, I couldn't understand your question."


# Streamlit chatbot UI

# # Streamlit chatbot UI
# st.title("Nebula9.ai Chatbot")
# st.write("Ask any question about Nebula9.ai or its services.")

# # Display chat history
# st.markdown("### Conversation")
# for message in st.session_state.chat_history:
#     if isinstance(message, HumanMessage):
#         st.markdown(f"**You:** {message.content}")
#     elif isinstance(message, AIMessage):
#         st.markdown(f"**Chatbot:** {message.content}")

# # Input box for user query at the bottom
# user_input = st.text_input("Your Question", "", key="user_input", label_visibility="collapsed")
# if user_input:
#     with st.spinner("Processing..."):
#         try:
#             chatbot_response = handle_query(user_input)
#             st.write(f"**Chatbot:** {chatbot_response}")
#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")



from streamlit_chat import message

st.title("Nebula9.ai Chatbot")
st.write("Ask any question about Nebula9.ai or its services.")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.markdown("### Conversation")
for chat in st.session_state.chat_history:
    if isinstance(chat, HumanMessage):
        st.chat_message("user").write(chat.content)
    elif isinstance(chat, AIMessage):
        st.chat_message("assistant").write(chat.content)


if user_input := st.chat_input("Your Question:"):
    
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.chat_message("user").write(user_input)

    with st.spinner("Processing..."):
        try:
            
            chatbot_response = handle_query(user_input)  
            st.session_state.chat_history.append(AIMessage(content=chatbot_response))
            st.chat_message("assistant").write(chatbot_response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")