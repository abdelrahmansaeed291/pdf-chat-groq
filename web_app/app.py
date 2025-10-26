import os, tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Load environment
load_dotenv()

st.set_page_config(page_title="Chat with Your PDF", page_icon="📄")
st.title("📄 Chat with Your PDF")

# Sidebar options
with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Groq model",
        ["llama-3.3-70b-versatile","llama-3.1-8b-instant" ],
        index=1
    )
    top_k = st.slider("Retriever top_k", 2, 10, 4)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_pdf_key" not in st.session_state:
    st.session_state.current_pdf_key = None

uploaded = st.file_uploader("Upload your PDF", type=["pdf"])

def build_chain(pdf_path: str):
    """Load, chunk, embed, and build a conversational chain."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )
    vectordb = Chroma.from_documents(chunks, embedding=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model=model,
        temperature=0
    )

    prompt = PromptTemplate.from_template(
        """You are a helpful assistant answering questions about the uploaded PDF.
Use ONLY the provided context to answer. If unsure, say you don't know.
Be concise and cite page numbers when possible.

Chat history:
{chat_history}

Context:
{context}

Question: {question}
Answer:"""
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain

# Rebuild chain when new PDF uploaded
if uploaded:
    pdf_key = f"{uploaded.name}:{uploaded.size}"
    if pdf_key != st.session_state.current_pdf_key:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            pdf_path = tmp.name
        with st.spinner("Indexing your PDF..."):
            st.session_state.chain = build_chain(pdf_path)
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.current_pdf_key = pdf_key
        st.success(f"PDF '{uploaded.name}' loaded successfully ✅")

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask a question about your PDF...")

if user_input:
    if not st.session_state.chain:
        st.warning("Please upload a PDF first.")
        st.stop()

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.chain({
                "question": user_input,
                "chat_history": st.session_state.chat_history
            })
        answer = result["answer"]
        st.markdown(answer)

        # Show supporting text chunks
        with st.expander("Show supporting context"):
            for i, doc in enumerate(result.get("source_documents", []), start=1):
                st.markdown(f"**Chunk {i} (page {doc.metadata.get('page', 'N/A')})**")
                st.write(doc.page_content)
                st.write("---")

    # Store conversation
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.append((user_input, answer))
