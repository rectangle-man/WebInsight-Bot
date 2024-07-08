from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import FAISS 
import pickle
import os
import streamlit as st
from llm import llm
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

HUGGINGFACEHUB_API_TOKEN = ""

st.title("WebInsights")

st.sidebar.title("Article URLs")

urls=[]

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URL")
file_path = "faiss_store_huggingface.pkl"

main_placefolder = st.empty()

if process_url_clicked:
    
    #Loading Data
    main_placefolder.text("Data Loading")
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    #Splitting the Data
    main_placefolder.text("Data Splitting")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size = 100,
        chunk_overlap = 1
    )

    docs = text_splitter.split_documents(data)

    #Creating Embeddings
    main_placefolder.text("Building Embedding Vectors")
    embeddings = HuggingFaceHubEmbeddings(huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN)
    vector_store_huggingface = FAISS.from_documents(docs, embeddings)

    #Saving FAISS Index to Pickle File
    with open(file_path, "wb") as f:
        pickle.dump(vector_store_huggingface,f)

question = main_placefolder.text_input("query: ")

if question:
    if os.path.exists(file_path):
        with open(file_path,"rb") as f:
            vector_store = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever = vector_store.as_retriever())
            result = chain({"question": question}, return_only_outputs=True)
            print(result)
            st.header("Answer")
            st.write(result["answer"] + result["sources"])
            

    

    
