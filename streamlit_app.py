import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
import os

# Loading API key
os.environ["OPENAI_API_KEY"] = st.secrets["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"]
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
