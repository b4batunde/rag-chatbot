import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
import os

# Loading API key
os.environ["OPENAI_API_KEY"] = st.secrets["openai.AuthenticationError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/rag-chatbot/streamlit_app.py", line 33, in <module>
    response = qa_chain.run(user_input)
File "/home/adminuser/venv/lib/python3.13/site-packages/langchain_core/_api/deprecation.py", line 191, in warning_emitting_wrapper
    return wrapped(*args, **kwargs)
File "/home/adminuser/venv/lib/python3.13/site-packages/langchain/chains/base.py", line 603, in run
    return self(args[0], callbacks=callbacks, tags=tags, metadata=metadata)[
           ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/langchain_core/_api/deprecation.py", line 191, in warning_emitting_wrapper
    return wrapped(*args, **kwargs)
File "/home/adminuser/venv/lib/python3.13/site-packages/langchain/chains/base.py", line 386, in __call__
    return self.invoke(
           ~~~~~~~~~~~^
        inputs,
        ^^^^^^^
    ...<2 lines>...
        include_run_info=include_run_info,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/langchain/chains/base.py", line 167, in invoke
    raise e
File "/home/adminuser/venv/lib/python3.13/site-packages/langchain/chains/base.py", line 157, in invoke
    self._call(inputs, run_manager=run_manager)
    ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/langchain/chains/retrieval_qa/base.py", line 151, in _call
    docs = self._get_docs(question, run_manager=_run_manager)
File "/home/adminuser/venv/lib/python3.13/site-packages/langchain/chains/retrieval_qa/base.py", line 271, in _get_docs
    return self.retriever.invoke(
           ~~~~~~~~~~~~~~~~~~~~~^
        question, config={"callbacks": run_manager.get_child()}
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/langchain_core/retrievers.py", line 258, in invoke
    result = self._get_relevant_documents(
        input, run_manager=run_manager, **_kwargs
    )
File "/home/adminuser/venv/lib/python3.13/site-packages/langchain_core/vectorstores/base.py", line 1079, in _get_relevant_documents
    docs = self.vectorstore.similarity_search(query, **_kwargs)
File "/home/adminuser/venv/lib/python3.13/site-packages/langchain_community/vectorstores/faiss.py", line 643, in similarity_search
    docs_and_scores = self.similarity_search_with_score(
        query, k, filter=filter, fetch_k=fetch_k, **kwargs
    )
File "/home/adminuser/venv/lib/python3.13/site-packages/langchain_community/vectorstores/faiss.py", line 515, in similarity_search_with_score
    embedding = self._embed_query(query)
File "/home/adminuser/venv/lib/python3.13/site-packages/langchain_community/vectorstores/faiss.py", line 266, in _embed_query
    return self.embedding_function.embed_query(text)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/langchain_community/embeddings/openai.py", line 704, in embed_query
    return self.embed_documents([text])[0]
           ~~~~~~~~~~~~~~~~~~~~^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/langchain_community/embeddings/openai.py", line 671, in embed_documents
    return self._get_len_safe_embeddings(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        texts, engine=engine, chunk_size=chunk_size
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/langchain_community/embeddings/openai.py", line 497, in _get_len_safe_embeddings
    response = embed_with_retry(
        self,
        input=tokens[i : i + _chunk_size],
        **self._invocation_params,
    )
File "/home/adminuser/venv/lib/python3.13/site-packages/langchain_community/embeddings/openai.py", line 120, in embed_with_retry
    return embeddings.client.create(**kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/openai/resources/embeddings.py", line 129, in create
    return self._post(
           ~~~~~~~~~~^
        "/embeddings",
        ^^^^^^^^^^^^^^
    ...<8 lines>...
        cast_to=CreateEmbeddingResponse,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/openai/_base_client.py", line 1239, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
                           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/openai/_base_client.py", line 1034, in request
    raise self._make_status_error_from_response(err.response) from None"]
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
