import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document  # Import the Document class
import os

DATA_PATH = 'data/'  # Folder containing your CSV files
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Function to ingest CSV into FAISS
def ingest_csv_to_faiss(csv_file):
    if not os.path.exists(csv_file):
        print(f"Error: The file {csv_file} does not exist.")
        return
    
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    # Combine all text columns or select specific columns
    documents = []
    for _, row in df.iterrows():
        text = " ".join(row.astype(str))  # Combine all columns into one string
        documents.append(Document(page_content=text))  # Wrap in a Document object
    
    # Debug: Check the type and content of the documents
    print(f"Loaded {len(documents)} documents.")
    print(f"First document: {documents[0]}")

    # Split the text into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)  # Use split_documents
    
    # Debug: Check the type and content of the split texts
    print(f"Split into {len(texts)} chunks.")
    print(f"First chunk: {texts[0]}")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', 
                                       model_kwargs={'device': 'cpu'})
    print("Creating embeddings and building FAISS index...")
    
    # Create FAISS database
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"FAISS database saved to {DB_FAISS_PATH}.")

# Create vector database
def create_vector_db():
    # Ingest CSV file(s)
    csv_files = [
        os.path.join(DATA_PATH, "Bhagwad_Gita_Verses_English_Questions.csv"),
        os.path.join(DATA_PATH, "Patanjali_Yoga_Sutras_Verses_English_Questions.csv")
    ]
    for csv_file in csv_files:
        print(f"Processing {csv_file}...")
        ingest_csv_to_faiss(csv_file)

if __name__ == "__main__":
    create_vector_db()
