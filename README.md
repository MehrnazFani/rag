# Retrieval-Augmented Generation (RAG)

## Environment setup

create a conda environment and activate it

```bash
conda create -n rag-env python=3.10
conda activate rag-env
```
### method 1:
install the following packages in your conda environment
```bash
pip install langchain
pip install langchain-community
pip install langchain-ollama
pip install faiss-cpu
pip install tiktoken
pip install -U langchain-ollama
```

## method 2:
``bash
pip install -r requirements.txt 
```

## Download Ollama and install
[https://ollama.com/download](https://ollama.com/download)

## Pull Ollama models by running
```bash
ollama pull llama3 # LLM for answering questions
ollama pull nomic-embed-text # Embedding model for vector search
```

