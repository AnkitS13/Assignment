from utils import store_docs, get_response, get_chroma_client
import os

os.environ["GROQ_API_KEY"] = "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"

store_docs("https://nebula9.ai/")

# Seeing how the data looks like in the vector store

vector_store = get_chroma_client()
lis = vector_store.get(include=['embeddings','metadatas','documents'])
print("Number of documents:",len(lis['documents']))
print("Content:\n",lis['documents'][1])
print("\n\nMetadata:", lis['metadatas'][1])
print("\n\nEmbeddings:", "Length:", len(lis['embeddings'][1]),"\nEmbedding Vector:",lis['embeddings'][1])

# Setting up organization information

organization_name = "Nebula9.ai"
organization_info = "We offer a diverse range of services that form the backbone of our AI solutions. From Artificial Intelligence & Machine Learning to Quality Assurance and Cloud Services, we provide end-to-end solutions designed for your success."
contact_info = """Email: info@nebula9.ai
India Phone: +91 9999032126
International Phone: +1 (412) 568-3901.
Book a consultation: https://nebula9.ai/book-a-free-consultation/
Get In Touch: https://nebula9.ai/contact-us//"""


# Get response

response = get_response("What is the company about ?", organization_name, organization_info, contact_info)
print("Answer:", response)




