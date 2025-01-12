from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Set the path to your vector store
vectorstore_path = 'vectorstore/db_faiss/Bhagwad_Gita_Verses_English_Questions'  # Update with your actual path

# Initialize the embeddings (same model used for creating the vector store)
embeddings = HuggingFaceEmbeddings()

# Load the FAISS vector store with embeddings, allowing dangerous deserialization
vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

# Perform a test query to verify it's working
query = "What is the writer of bagavadhgita?"
results = vectorstore.similarity_search(query, k=3)  # Change 'k' to the number of results you want

# Print the results
for result in results:
    print(result)
