from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
import os

# Step 1: Load documents from ./docs/*.txt
docs_path = "./docs"
loaders = [TextLoader(os.path.join(docs_path, f)) for f in os.listdir(docs_path) if f.endswith(".txt")]
documents = []
for loader in loaders:
    documents.extend(loader.load())


# Step 2: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)


# Step 3: Generate embeddings using Ollama
embedding = OllamaEmbeddings(model="nomic-embed-text")  # Make sure you have this model pulled with ollama
db = FAISS.from_documents(texts, embedding)


# Step 4: Initialize retriever and Ollama LLM
retriever = db.as_retriever()
llm = OllamaLLM(model="llama3")  # You can replace with "mistral" or any other model you've pulled


# Step 5: Setup QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)


# Step 6: Ask a question
query = "What is RAG and how does it work?"

result = qa_chain.invoke({"query": query})

# Print answer and sources
print("ANSWER:")
print(result["result"])
print("\nSOURCES:")
for doc in result["source_documents"]:
    print(doc.metadata["source"])
