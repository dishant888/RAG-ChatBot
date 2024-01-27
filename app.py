import os
import pickle
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

def main():

    if not os.path.exists("embedded_docs"):
        os.makedirs("embedded_docs")

    st.header("RAG based Chatbot")

    # File upload
    pdf = st.file_uploader(label = "Choose a PDF file", type = "PDF", accept_multiple_files = False)

    # Read the PDF
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len,
        )

        vector_store_name = pdf.name[:-4]
        
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
        
        query = st.text_input("Ask any question related to the PDF....")

        if query:
            relevant_docs = vector_store.similarity_search(query = query)
            st.write(relevant_docs)
        


if __name__ == "__main__":
    main()