import streamlit as st
import requests
import json
import logging
from io import BytesIO
from collections import Counter
from datetime import datetime

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

st.set_page_config(layout="wide", page_title="RAG Application", initial_sidebar_state="expanded")

# Initialize session state
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:8000/api"
if 'uploaded_docs' not in st.session_state:
    st.session_state.uploaded_docs = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'current_response' not in st.session_state:
    st.session_state.current_response = ""
if 'selected_doc_ids' not in st.session_state:
    st.session_state.selected_doc_ids = []
if 'query_stats' not in st.session_state:
    st.session_state.query_stats = {'times': [], 'doc_hits': Counter(), 'total_tokens': 0}

# Helper functions
def fetch_documents():
    try:
        response = requests.get(f"{st.session_state.api_url}/documents", timeout=10)
        response.raise_for_status()
        docs = response.json()
        logger.debug(f"Documents API response: {docs}")
        # Calculate corpus size (approximate, since content isn't returned)
        corpus_size = sum(doc.get("content_length", 0) for doc in docs)
        return docs, corpus_size
    except Exception as e:
        logger.error(f"Failed to fetch documents: {str(e)}")
        st.error(f"Error fetching documents: {str(e)}")
        return [], 0

def upload_document(file):
    try:
        files = {"file": (file.name, BytesIO(file.read()), file.type)}
        response = requests.post(f"{st.session_state.api_url}/documents/upload", files=files, timeout=30)
        response.raise_for_status()
        doc_id = response.json().get("document_id")
        logger.info(f"Uploaded document: {file.name}, ID: {doc_id}")
        return doc_id
    except Exception as e:
        logger.error(f"Failed to upload document: {str(e)}")
        st.error(f"Error uploading document: {str(e)}")
        return None

def delete_document(doc_id):
    try:
        response = requests.delete(f"{st.session_state.api_url}/documents/{doc_id}", timeout=10)
        response.raise_for_status()
        logger.info(f"Deleted document ID: {doc_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete document: {str(e)}")
        st.error(f"Error deleting document: {str(e)}")
        return False

# Page navigation
page = st.sidebar.selectbox("Select Page", ["Chat", "Documents", "Document Management", "Stats"])

# Chat Page
if page == "Chat":
    st.title("RAG Chat")
    st.markdown("Ask questions based on uploaded documents.")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.write(f"- {source['filename']} (Chunk {source['chunk_index']}): {source['content'][:100]}...")
                st.caption(f"Tokens: {message.get('tokens_used', 0)}, Time: {message.get('response_time', 0):.2f}s")

    # Document selection
    docs, _ = fetch_documents()
    doc_options = {doc["filename"]: doc["id"] for doc in docs}
    selected_docs = st.multiselect("Select documents to query", options=doc_options.keys(), default=[])
    st.session_state.selected_doc_ids = [doc_options[doc] for doc in selected_docs]

    # Chat input
    question = st.chat_input("Ask a question")
    if question:
        st.session_state.current_question = question
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.current_response = ""
        st.session_state.current_sources = []
        with st.chat_message("assistant"):
            response_container = st.empty()
            try:
                payload = {"question": question, "doc_ids": st.session_state.selected_doc_ids}
                with requests.post(f"{st.session_state.api_url}/query_stream", json=payload, stream=True, timeout=30) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line.decode('utf-8'))
                            logger.debug(f"Chunk received: {chunk}")
                            if chunk["type"] == "content":
                                st.session_state.current_response += chunk["chunk"]
                                response_container.markdown(st.session_state.current_response)
                            elif chunk["type"] == "complete":
                                st.session_state.current_sources = chunk["sources"]
                                st.session_state.query_stats['times'].append(chunk["response_time"])
                                st.session_state.query_stats['total_tokens'] += chunk["tokens_used"]
                                for source in chunk["sources"]:
                                    st.session_state.query_stats['doc_hits'][source["filename"]] += 1
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": st.session_state.current_response,
                                    "sources": chunk["sources"],
                                    "response_time": chunk["response_time"],
                                    "tokens_used": chunk["tokens_used"]
                                })
                            elif chunk["type"] == "error":
                                st.error(f"Query failed: {chunk['message']}")
            except Exception as e:
                logger.error(f"Streaming query failed: {str(e)}")
                st.error(f"Error: {str(e)}")

# Documents Page
elif page == "Documents":
    st.title("Documents")
    docs, corpus_size = fetch_documents()
    if docs:
        st.write("### Document List")
        st.metric("Corpus Size", f"{corpus_size:,} characters (approximate)")
        for doc in docs:
            st.write(f"- {doc['filename']} (ID: {doc['id']}, Created: {doc['created_at']})")
    else:
        st.write("No documents found.")

# Document Management Page
elif page == "Document Management":
    st.title("Document Management")
    st.markdown("Upload or delete documents.")

    # Upload section
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
    if uploaded_file:
        if st.button("Upload"):
            with st.spinner("Uploading..."):
                doc_id = upload_document(uploaded_file)
                if doc_id:
                    st.success(f"Uploaded {uploaded_file.name} with ID: {doc_id}")
                    st.session_state.uploaded_docs.append({"filename": uploaded_file.name, "id": doc_id})

    # Document list with delete option
    st.subheader("Manage Documents")
    docs, _ = fetch_documents()
    if docs:
        for doc in docs:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{doc['filename']} (ID: {doc['id']}, Created: {doc['created_at']})")
            with col2:
                if st.button("Delete", key=f"delete_{doc['id']}"):
                    if delete_document(doc['id']):
                        st.success(f"Deleted {doc['filename']}")
                        st.session_state.uploaded_docs = [d for d in st.session_state.uploaded_docs if d['id'] != doc['id']]
                        st.rerun()
    else:
        st.write("No documents available.")

# Stats Page
elif page == "Stats":
    st.title("Stats")
    docs, corpus_size = fetch_documents()
    st.metric("Corpus Size", f"{corpus_size:,} characters (approximate)")
    st.metric("Total Tokens Used", st.session_state.query_stats['total_tokens'])
    if st.session_state.query_stats['times']:
        avg_time = sum(st.session_state.query_stats['times']) / len(st.session_state.query_stats['times'])
        st.metric("Average Response Time", f"{avg_time:.2f} seconds")
    else:
        st.metric("Average Response Time", "No queries yet")

    st.subheader("Most Often Hit Documents")
    if st.session_state.query_stats['doc_hits']:
        top_docs = st.session_state.query_stats['doc_hits'].most_common(5)
        data = {doc: hits for doc, hits in top_docs}
        st.bar_chart(data)
        for doc, hits in top_docs:
            st.write(f"- {doc}: {hits} hits")
    else:
        st.write("No document hits yet.")

    st.subheader("Query History")
    if st.session_state.chat_history:
        for i, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                st.write(f"**Q{i//2 + 1}**: {msg['content']}")
                if i + 1 < len(st.session_state.chat_history) and st.session_state.chat_history[i + 1]["role"] == "assistant":
                    response = st.session_state.chat_history[i + 1]
                    st.write(f"**Response**: {response['content'][:100]}...")
                    st.write(f"Tokens: {response['tokens_used']}, Time: {response['response_time']:.2f}s")
    else:
        st.write("No queries yet.")