from langchain_community.vectorstores import Chroma
from text_to_doc import process_crawled_data
from web_crawler2 import crawl_website
from prompt import get_prompt
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Initialize Chroma Client
def get_chroma_client():
    """
    Returns a Chroma vector store instance.

    Returns:
        langchain.vectorstores.chroma.Chroma: ChromaDB vector store instance.
    """
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(
        collection_name="website_data",
        embedding_function=embedding_function,
        # persist_directory="data/chroma"  # Uncomment to persist data
    )

# Store Documents into Vector Store
def store_docs(base_url, max_pages=10):
    """
    Crawls a website, processes crawled data into document chunks, and stores them in a vector store.

    Args:
        base_url (str): The starting URL of the website.
        max_pages (int): Maximum number of pages to crawl.
    """
    # Crawl the website
    pages_data = crawl_website(base_url, max_pages=max_pages)

    # Process crawled data into document chunks
    docs = process_crawled_data(pages_data)

    # Add documents to Chroma vector store
    vector_store = get_chroma_client()
    vector_store.add_documents(docs)

    print(f"Stored {len(docs)} document chunks from {base_url}.")
    # vector_store.persist()  # Uncomment to save data persistently

# Create Conversational Chain
def make_chain():
    """
    Creates a conversational retrieval chain using a vector store and Groq model.

    Returns:
        langchain.chains.ConversationalRetrievalChain: ConversationalRetrievalChain instance.
    """
    model = ChatGroq(
        model_name="mixtral-8x7b-32768",
        temperature=0.0,
        verbose=True
    )
    vector_store = get_chroma_client()
    prompt = get_prompt()

    retriever = vector_store.as_retriever(search_type="mmr", verbose=True)

    chain = ConversationalRetrievalChain.from_llm(
        model,
        retriever=retriever,
        return_source_documents=False,
        combine_docs_chain_kwargs=dict(prompt=prompt),
        verbose=False,
        rephrase_question=False,
    )
    return chain

# Generate Response with Chat History
chat_history = []  # Global variable to maintain conversation context

def get_response(question, organization_name, organization_info, contact_info, relevant_docs=None):
    """
    Generates a response while maintaining conversational context.
    Handles both general and company-specific queries.

    Args:
        question (str): The user's query.
        organization_name (str): Name of the organization.
        organization_info (str): Info about the organization.
        contact_info (str): Contact info for the organization.
        relevant_docs (List[Document], optional): Relevant documents for company-specific context.

    Returns:
        str: The chatbot's response.
    """
    global chat_history
    chain = make_chain()

    # Decide the context: Use relevant_docs if available
    if relevant_docs:
        context = " ".join([doc.page_content for doc in relevant_docs])
        prompt = f"Based on the following company-specific context, respond to the question: {question}\n\nContext: {context}"
    else:
        prompt = f"Question: {question}\n\nProvide a helpful and engaging response."

    # Generate response
    response = chain({
        "question": prompt,
        "chat_history": chat_history,
        "organization_name": organization_name,
        "contact_info": contact_info,
        "organization_info": organization_info
    })

    # Update chat history
    chat_history.append((question, response['answer']))
    return response['answer']





