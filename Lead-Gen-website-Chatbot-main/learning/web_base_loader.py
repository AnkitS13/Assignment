import os
import bs4
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

# Set the API key directly
os.environ["GROQ_API_KEY"] = "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"


# Initialize the LLM with the desired model
llm = ChatGroq(model="llama3-8b-8192")

# 1. Load, chunk and index the contents of the blog to create a retriever.
loader = WebBaseLoader(
    web_paths=("https://nebula9.ai/services/gen-ai/","https://nebula9.ai/services/cloud-services/"),
    #bs_kwargs=dict(
        #parse_only=bs4.SoupStrainer(
            #class_=("post-content", "post-title", "post-header")
        #)
    #),
)
docs = loader.load()

# 2. Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 3. Use Hugging Face embeddings 
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Initialize ChromaDB and store the embeddings
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding_model,
    persist_directory="chroma_db"  
)

# 5. Create a retriever
retriever = vectorstore.as_retriever()

# 6. Persist the database to reuse it later
#vectorstore.persist()

# 6. Contextualize questions with history-aware retriever
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

# 7. Define QA chain
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use five sentences maximum and keep the "
    "answer concise."
    "\n\n"
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

# 8. Create final RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 9. Maintain chat history and ask questions
chat_history = []

# First question
question_1 = "What are the services offered ? "
response_1 = rag_chain.invoke({"input": question_1, "chat_history": chat_history})
chat_history.extend(
    [
        HumanMessage(content=question_1),
        AIMessage(content=response_1["answer"]),
    ]
)
print("Q1:", question_1)
print("A1:", response_1["answer"])

# Follow-up question
question_2 = "What are cloud solutions services ?"
response_2 = rag_chain.invoke({"input": question_2, "chat_history": chat_history})
chat_history.extend(
    [
        HumanMessage(content=question_2),
        AIMessage(content=response_2["answer"]),
    ]
)
print("Q2:", question_2)
print("A2:", response_2["answer"])