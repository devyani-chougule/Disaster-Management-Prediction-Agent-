from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load cyclone knowledge

loader = TextLoader(
    r"C:\Users\Devyani Chougule\Desktop\NLP (Disaster Managment)\cycloneinfo.txt"
)

documents = loader.load()


# Create embeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create vector database

db = Chroma.from_documents(
    documents,
    embedding_model
)


# Test retrieval

query = "What safety actions should people take during cyclone?"

results = db.similarity_search(
    query,
    k=2
)


for doc in results:
    print(doc.page_content)