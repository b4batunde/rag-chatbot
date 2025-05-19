import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
import os

# Loading API key
os.environ["OPENAI_API_KEY"] = ["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"]

@st.cache_resource
def load_qa_chain():
    loader = TextLoader("data/python_guide.txt")
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

qa_chain = load_qa_chain()

# UI
st.title("Python RAG Chatbot")
st.markdown("Ask a question about Python basics:")

user_input = st.text_input("You:")

if user_input:
    with st.spinner("Thinking..."):
        response = qa_chain.run(user_input)
    st.success("Bot says:")
    st.write(response)
