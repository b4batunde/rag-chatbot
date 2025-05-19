from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
import os

# OpenAI API Key
os.environ["OPENAI_API_KEY"] = "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"

# Loading data
loader = TextLoader("data/python_guide.txt")
documents = loader.load()

# Splitting text into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(documents)

# Vector store with embeddings
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embeddings)

# Create QA chain
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

# Chat loop
if __name__ == "__main__":
    print("Welcome to the Python Guide Chatbot! Type your question below.")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = qa_chain.run(query)
        print("Bot:", response)
