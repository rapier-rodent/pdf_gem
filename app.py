import os
from dotenv import load_dotenv
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# Initialize SessionState for persistent storage
if "qa" not in st.session_state:
    st.session_state.qa = None

# Sidebar for uploading PDF
st.sidebar.title("PDF Analyzer")
pdf_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

# Function to process the PDF and create QA system
def process_pdf(pdf_file):
    # Load and split the PDF into chunks
    loader = PyPDFLoader(pdf_file.name)  
    documents = loader.load_and_split()

    # Split into smaller chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    docs = text_splitter.split_documents(documents)

    # Create embeddings and store in a vector database
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings) 
    
    # Create the question-answering system
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=vectorstore.as_retriever()
    )
    return qa

# Main Streamlit app interaction
if pdf_file is not None:
    # Process the PDF only once on upload
    if st.session_state.qa is None:
        with st.spinner("Processing your PDF..."):
            st.session_state.qa = process_pdf(pdf_file)

    # Allow user to ask questions
    st.header("Ask a question about your PDF:")
    question = st.text_input("")
    if st.button("Get Answer"):
        if question:
            with st.spinner("Thinking..."):
                answer = st.session_state.qa.run(question) 
                st.write(answer)
        else:
            st.warning("Please enter a question.")
else:
    st.info("Upload a PDF to get started.")
