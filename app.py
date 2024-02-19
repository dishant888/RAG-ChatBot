import os
import pickle
import requests
import json
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

def main():
    # Create a folder to store embeddings
    if not os.path.exists("embedded_docs"):
        os.makedirs("embedded_docs")

    st.header("RAG based Chatbot")

    # File upload
    pdf = st.file_uploader(label = "Choose a PDF file", type = "PDF", accept_multiple_files = False)

    # Read the PDF
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""

        # Extracting text from the PDF
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Create an Instance of Text Splitter to create overlaping chunks of the text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len,
        )

        # Get the name of the PDF document
        vector_store_name = pdf.name[:-4]
        
        # Generate embeddings (if it doesn't exist), else load from the disk
        if os.path.exists(f"embedded_docs/{vector_store_name}.pkl"):
            with st.spinner("Loading embeddings from disk!"):
                with open(f"embedded_docs/{vector_store_name}.pkl", "rb") as f:
                    vector_store = pickle.load(f)
        else:
            with st.spinner("Generating embeddings .... hold on tight ... it might take some time!"):
                chunks = text_splitter.split_text(text)
                embedder = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
                vector_store = FAISS.from_texts(chunks, embedder)
                with open(f"embedded_docs/{vector_store_name}.pkl", "wb") as f:
                    pickle.dump(vector_store, f)
        st.balloons()

        
        # Display a input box for user to ener query
        question = st.text_input("Ask any question related to the PDF....").strip()

        if question:

            # Generate response to the user query
            with st.spinner("Generating answer..."):

                # Get top 3 matching documents
                relevant_docs = vector_store.similarity_search(question, 3)
                context = "\n\n".join([d.page_content for d in relevant_docs])
                
                # Prompt template for the LLM
                template = f"Answer the question based only on the following context: \"{context}.\" Question: {question}"
                
                # API settings
                headers = {"Authorization": st.secrets["API_KEY"]}
                url = "https://api.edenai.run/v2/text/chat"

                payload = {
                    "providers": "openai",
                    "text": template,
                    "chatbot_global_action": "You are an AI assistant that answers questions based on the documents uploaded by the user. Only use the context provided to you to answer the question. If you can't find the answer in the provided context, tell the user that you can't find the answer. Do not directly copy the context when generation the answer.",
                    "previous_history": [],
                    "temperature": 0.0,
                    "max_tokens": 300,
                    "fallback_providers": ""
                }
                
                # Make request to the API
                response = requests.post(url, json = payload, headers = headers)
                result = json.loads(response.text)

                # Display results to the user
                st.write(result['openai']['generated_text'])
        
if __name__ == "__main__":
    main()