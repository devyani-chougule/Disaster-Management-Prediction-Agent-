from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings



def get_guidelines(query):


    loader = TextLoader(
        r"C:\Users\Devyani Chougule\Desktop\NLP project\data\disaster_guidelines.txt"
    )


    documents = loader.load()



    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )



    db = Chroma.from_documents(
        documents,
        embeddings
    )



    results = db.similarity_search(
    query,
    k=3
)


    context = "\n".join(
        [doc.page_content for doc in results]
    )


    return context