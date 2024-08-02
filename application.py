import streamlit as st
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import tempfile

# Set OpenAI API key
openai_api_key = st.secrets["openai"]["api_key"]
os.environ["OPENAI_API_KEY"] = openai_api_key

def load_db(files, chain_type, k):
    documents = []
    for file in files:
        loader = PyPDFLoader(file)
        documents.extend(loader.load())
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # Define embedding
    embeddings = OpenAIEmbeddings()
    # Create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # Define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # Create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa 

# Streamlit UI
st.title('ChatWithYourData_Bot')

# File uploader
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    file_paths = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_paths.append(tmp_file.name)

    st.write(f"Loaded Files: {[file.name for file in uploaded_files]}")
    qa = load_db(file_paths, "stuff", 4)

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        st.write(f"**User:** {question}")
        st.write(f"**ChatBot:** {answer}")

    query = st.text_input('Enter your question:', key='query', on_change=lambda: st.session_state.chat_history.append((st.session_state.query, qa({"question": st.session_state.query, "chat_history": st.session_state.chat_history})["answer"])))

    # Scroll to the bottom
    st.write(f'<div id="bottom"></div>', unsafe_allow_html=True)
    st.markdown('<script>window.scrollTo(0, document.body.scrollHeight);</script>', unsafe_allow_html=True)
else:
    st.write("Please upload PDF files to start.")
