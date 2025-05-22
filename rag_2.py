import os
from glob import glob
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

def load_documents_from_paths(file_paths):
    all_documents = []
    for path in file_paths:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".pdf":
                loader = PyMuPDFLoader(path)
            elif ext == ".txt":
                loader = TextLoader(path)
            else:
                print(f"Skipping unsupported file: {path}")
                continue
            docs = loader.load()
            all_documents.extend(docs)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
    return all_documents

def get_file_list_from_directory(directory, extensions=[".pdf", ".txt"]):
    file_paths = []
    for ext in extensions:
        file_paths.extend(glob(os.path.join(directory, f"*{ext}")))
    return file_paths

def main(input_dir, query):
    file_paths = get_file_list_from_directory(input_dir)
    if not file_paths:
        print("No valid files found.")
        return

    print(f"Loading {len(file_paths)} files...")
    documents = load_documents_from_paths(file_paths)

    # Chunk the documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Embedding
    print("Creating vector store...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # LLM using OllamaLLM
    llm = OllamaLLM(model="llama3")

    # Retrieval-QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    # Ask your question
    print(f"\nQuestion: {query}")
    result = qa_chain.invoke({"query": query})

    # Output
    print("\nAnswer:\n", result["result"])
    print("\nSources:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("source", "[unknown]"))

if __name__ == "__main__":
    # Set this to your folder and question
    input_directory = "./docs"
    user_query = "list the positive effects of sport"

    main(input_directory, user_query)
