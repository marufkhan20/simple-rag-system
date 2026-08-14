from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistant_directory = "db/chroma_db"

# Load embeddings and vector store
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistant_directory,
    embedding_function=embeddings_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Search for relevant documents
query = "Which island does SpaceX lease for its launches in the Pacifix"

retriever = db.as_retriever(search_kwargs={"k": 3})

relevant_docs = retriever.invoke(query)

print(f"User query: {query}")

print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
