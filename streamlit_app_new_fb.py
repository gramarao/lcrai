import streamlit as st
import requests
import json
import logging
from io import BytesIO
from collections import Counter
from datetime import datetime
import uuid
import os

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/streamlit_app.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

st.set_page_config(layout="wide", page_title="RAG Application", initial_sidebar_state="expanded")

# Initialize session state with stable IDs
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:8000/api"
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
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
if 'current_sources' not in st.session_state:
    st.session_state.current_sources = []
if 'current_query_id' not in st.session_state:
    st.session_state.current_query_id = None

# Helper functions
def fetch_documents():
    logger.debug("Entering fetch_documents")
    try:
        response = requests.get(f"{st.session_state.api_url}/documents", timeout=10)
        response.raise_for_status()
        docs = response.json()
        logger.info(f"Fetched {len(docs)} documents")
        corpus_size = sum(doc.get("content_length", 0) for doc in docs)
        return docs, corpus_size
    except Exception as e:
        logger.error(f"Failed to fetch documents: {str(e)}")
        st.error(f"Error fetching documents: {str(e)}")
        return [], 0
    finally:
        logger.debug("Exiting fetch_documents")

def upload_document(file):
    logger.debug(f"Entering upload_document: {file.name}")
    try:
        files = {"file": (file.name, BytesIO(file.read()), file.type)}
        response = requests.post(f"{st.session_state.api_url}/documents/upload", files=files, timeout=120)
        response.raise_for_status()
        doc_id = response.json().get("document_id")
        logger.info(f"Uploaded document: {file.name}, ID: {doc_id}")
        return doc_id
    except Exception as e:
        logger.error(f"Failed to upload document: {str(e)}")
        st.error(f"Error uploading document: {str(e)}")
        return None
    finally:
        logger.debug("Exiting upload_document")

def delete_document(doc_id):
    logger.debug(f"Entering delete_document: {doc_id}")
    try:
        response = requests.delete(f"{st.session_state.api_url}/documents/{doc_id}", timeout=10)
        response.raise_for_status()
        logger.info(f"Deleted document ID: {doc_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete document: {str(e)}")
        st.error(f"Error deleting document: {str(e)}")
        return False
    finally:
        logger.debug("Exiting delete_document")

# FIXED: Handle 'id' from /api/query_stream and output 'chunk_id' for /api/feedback
def validate_sources(sources):
    logger.info(f"Validating sources: {json.dumps(sources, indent=2) if sources else 'None'}")
    valid_sources = []
    
    if not sources:
        logger.warning("No sources provided for validation")
        return valid_sources
    
    for source in sources:
        if not isinstance(source, dict):
            logger.error(f"Invalid source format: {source}")
            continue
        if 'document_id' not in source or 'id' not in source:
            logger.error(f"Missing document_id or id in source: {source}")
            continue
        
        try:
            doc_id = str(source['document_id'])
            chunk_id = str(source['id'])  # Get the chunk ID from 'id' field
            
            # Optional UUID normalization
            try:
                doc_id = str(uuid.UUID(doc_id))
            except ValueError:
                logger.debug(f"document_id is not a valid UUID, keeping as string: {doc_id}")
            
            try:
                chunk_id = str(uuid.UUID(chunk_id))
            except ValueError:
                logger.debug(f"chunk id is not a valid UUID, keeping as string: {chunk_id}")
            
            # FIXED: Output 'id' to match Pydantic model expectation
            valid_sources.append({
                "document_id": doc_id,
                "id": chunk_id,  # Use 'id' not 'chunk_id'
                "relevance_score": source.get("relevance_score", 1.0),
                "filename": source.get("filename", ""),
                "chunk_index": source.get("chunk_index", 0)
            })
            logger.debug(f"Validated source: document_id={doc_id}, id={chunk_id}")
            
        except Exception as e:
            logger.error(f"Error processing source {source}: {str(e)}")
            continue
    
    logger.info(f"Valid sources after validation: {len(valid_sources)} sources")
    return valid_sources

# Page navigation
page = st.sidebar.selectbox("Select Page", ["Chat", "Documents", "Document Management", "Stats"])

# Chat Page
if page == "Chat":
    st.title("RAG Chat")
    st.markdown("Ask questions based on uploaded documents.")
    logger.debug("Chat page loaded")

    # Debug info
    st.sidebar.text(f"Session ID: {st.session_state.session_id[:8]}...")
    
    # Debug button to log session state
    if st.sidebar.button("Debug Session State"):
        logger.debug(f"Session state keys: {list(st.session_state.keys())}")
        st.sidebar.success("Session state logged")

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.write(f"- **Document ID**: {source['document_id']}")
                        st.write(f"  **Chunk ID**: {source['id']}")  # FIXED: Use 'id' from /api/query_stream
                        st.write(f"  **File**: {source.get('filename', 'Unknown')} (Chunk {source.get('chunk_index', 'N/A')})")
                        st.write(f"  **Content**: {source.get('content', 'No content')[:300]}...")
                st.caption(f"Response Time: {message.get('response_time', 0):.2f}s, Tokens Used: {message.get('tokens_used', 0)}")

    # Document selection
    docs, _ = fetch_documents()
    doc_options = {doc["filename"]: doc["id"] for doc in docs}
    selected_docs = st.multiselect("Select documents to query", options=doc_options.keys(), default=[])
    st.session_state.selected_doc_ids = [doc_options[doc] for doc in selected_docs]
    logger.debug(f"Selected doc_ids: {st.session_state.selected_doc_ids}")

    # Chat input
    question = st.chat_input("Ask a question")
    if question:
        # Generate stable query ID
        st.session_state.current_query_id = str(uuid.uuid4())
        logger.info(f"Processing query: {question[:50]}...")
        st.session_state.current_question = question
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.current_response = ""
        st.session_state.current_sources = []
        with st.chat_message("assistant"):
            response_container = st.empty()
            logger.debug("Starting query stream")
            try:
                payload = {
                    "question": question,
                    "doc_ids": [str(doc_id) for doc_id in st.session_state.selected_doc_ids],
                    "query_id": st.session_state.current_query_id,
                    "session_id": st.session_state.session_id
                }
                logger.info(f"Sending query to /api/query_stream with payload: {json.dumps(payload, indent=2)}")
                with requests.post(f"{st.session_state.api_url}/query_stream", json=payload, stream=True, timeout=120) as response:
                    response.raise_for_status()
                    logger.debug("Query stream response started")
                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line.decode('utf-8'))
                            logger.debug(f"Received chunk: {json.dumps(chunk, indent=2)}")
                            if chunk["type"] == "content":
                                st.session_state.current_response += chunk["chunk"]
                                response_container.markdown(st.session_state.current_response)
                            elif chunk["type"] == "complete":
                                st.session_state.current_sources = chunk.get("sources", [])
                                logger.info(f"Query completed, received {len(st.session_state.current_sources)} sources")
                                logger.debug(f"Sources received: {json.dumps(st.session_state.current_sources, indent=2)}")
                                st.session_state.query_stats['times'].append(chunk["response_time"])
                                st.session_state.query_stats['total_tokens'] += chunk["tokens_used"]
                                for source in st.session_state.current_sources:
                                    st.session_state.query_stats['doc_hits'][source["filename"]] += 1
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": st.session_state.current_response,
                                    "question": st.session_state.current_question,  # Add this
                                    "sources": st.session_state.current_sources,
                                    "response_time": chunk["response_time"],
                                    "tokens_used": chunk["tokens_used"],
                                    "query_id": st.session_state.current_query_id,
                                    "completed": True,
                                    "feedback_submitted": False
                                })
                                # Display response
                                st.markdown("**Full Response**:")
                                st.markdown(st.session_state.current_response)
                                if st.session_state.current_sources:
                                    st.markdown("**Sources**:")
                                    for source in st.session_state.current_sources:
                                        st.write(f"- **Document ID**: {source['document_id']}")
                                        st.write(f"  **Chunk ID**: {source['id']}")  # FIXED: Use 'id' from /api/query_stream
                                        st.write(f"  **File**: {source['filename']} (Chunk {source['chunk_index']})")
                                        st.write(f"  **Content**: {source['content'][:300]}...")
                                else:
                                    st.warning("No sources returned for this query.")
                                    logger.warning("No sources returned for query")
                                st.caption(f"Response Time: {chunk['response_time']:.2f}s, Tokens Used: {chunk['tokens_used']}")
                            elif chunk["type"] == "error":
                                st.error(f"Query failed: {chunk['message']}")
                                logger.error(f"Query stream error: {chunk['message']}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Streaming query failed: {str(e)}")
            logger.debug("Query stream completed")

    # Render feedback form only for the last completed assistant message
    if st.session_state.chat_history:
        last_msg = st.session_state.chat_history[-1]
        if (last_msg.get("role") == "assistant" and 
            last_msg.get("completed") and 
            last_msg.get("query_id") and
            not last_msg.get("feedback_submitted")):
            form_key = f"feedback_{last_msg['query_id']}"
            logger.info(f"Rendering feedback form with key: {form_key}")
            with st.form(form_key):
                st.markdown("**Provide Feedback**")
                rating = st.slider("Rate this response (1-5)", 1, 5, 3, key=f"rating_{form_key}")
                thumbs_up = st.checkbox("Thumbs Up", key=f"thumbs_{form_key}")
                feedback_text = st.text_area("Comments (optional)", key=f"comment_{form_key}")
                submitted = st.form_submit_button("Submit Feedback")
                # In your streamlit_app.py, in the feedback submission section:
                if submitted:
                    logger.info("=== STREAMLIT FEEDBACK DEBUG ===")
                    try:
                        # Right before calling validate_sources, add:
                        logger.info("=== RAW SOURCES DEBUG ===")
                        raw_sources = last_msg.get("sources", [])
                        for i, source in enumerate(raw_sources):
                            logger.info(f"Raw source {i+1}: {json.dumps(source, indent=2, default=str)}")

                        valid_sources = validate_sources(last_msg.get("sources", []))
                        # After validate_sources:
                        logger.info("=== VALIDATED SOURCES DEBUG ===")
                        for i, source in enumerate(valid_sources):
                            logger.info(f"Validated source {i+1}: {json.dumps(source, indent=2, default=str)}")
                        logger.info(f"Validated {len(valid_sources)} sources")
                        
                        # Log each source for debugging
                        for i, source in enumerate(valid_sources):
                            logger.info(f"Source {i+1}: {json.dumps(source, indent=2)}")
                        
                        feedback_payload = {
                            "session_id": st.session_state.session_id,
                            "query_id": last_msg["query_id"],
                            "question": last_msg.get("question", st.session_state.current_question),
                            "response": last_msg.get("content", ""),
                            "rating": int(rating),  # Ensure it's an integer
                            "thumbs_up": bool(thumbs_up),  # Ensure it's a boolean
                            "feedback_text": feedback_text or "",  # Ensure it's not None
                            "response_time": float(last_msg.get("response_time", 0.0)),
                            "tokens_used": int(last_msg.get("tokens_used", 0)),
                            "sources_count": len(valid_sources),
                            "sources": valid_sources
                        }
                        
                        # Add quality scores if available
                        quality_scores = last_msg.get("quality_scores")
                        if quality_scores and isinstance(quality_scores, dict):
                            feedback_payload["quality_scores"] = {
                                "relevance": float(quality_scores.get("relevance", 0.0)),
                                "accuracy": float(quality_scores.get("accuracy", 0.0)),
                                "completeness": float(quality_scores.get("completeness", 0.0)),
                                "coherence": float(quality_scores.get("coherence", 0.0)),
                                "citation": float(quality_scores.get("citation", 0.0)),
                                "overall": float(quality_scores.get("overall", 0.0))
                            }
                            logger.info(f"Added quality scores: {feedback_payload['quality_scores']}")
                        
                        logger.info("=== FINAL PAYLOAD ===")
                        logger.info(json.dumps(feedback_payload, indent=2, default=str))
                        
                        logger.info("Sending feedback to API...")
                        feedback_response = requests.post(
                            f"{st.session_state.api_url}/feedback", 
                            json=feedback_payload, 
                            timeout=30,
                            headers={"Content-Type": "application/json"}
                        )
                        
                        logger.info(f"API Response Status: {feedback_response.status_code}")
                        logger.info(f"API Response Headers: {dict(feedback_response.headers)}")
                        
                        if feedback_response.status_code != 200:
                            logger.error(f"API Response Body: {feedback_response.text}")
                        
                        feedback_response.raise_for_status()
                        st.success("Feedback submitted successfully!")
                        logger.info("✅ Feedback submitted successfully")
                        last_msg["feedback_submitted"] = True
                        
                    except requests.exceptions.HTTPError as e:
                        logger.error("=== HTTP ERROR DETAILS ===")
                        err_resp = e.response
                        if err_resp:
                            logger.error(f"Status Code: {err_resp.status_code}")
                            logger.error(f"Response Headers: {dict(err_resp.headers)}")
                            try:
                                error_detail = err_resp.json()
                                logger.error(f"Error JSON: {json.dumps(error_detail, indent=2)}")
                                st.error(f"Feedback submission failed: {error_detail}")
                            except:
                                error_text = err_resp.text
                                logger.error(f"Error Text: {error_text}")
                                st.error(f"Feedback submission failed: {error_text}")
                        else:
                            st.error(f"HTTP Error: {str(e)}")
                        logger.error(f"Full HTTP error: {str(e)}")
                    except Exception as e:
                        logger.exception("Unexpected error in feedback submission")
                        st.error(f"Failed to submit feedback: {str(e)}")

# Documents Page
elif page == "Documents":
    st.title("Documents")
    docs, corpus_size = fetch_documents()
    if docs:
        st.write("### Document List")
        st.metric("Corpus Size", f"{corpus_size:,} characters")
        for doc in docs:
            st.write(f"- {doc['filename']} (ID: {doc['id']}, Created: {doc['created_at']})")
    else:
        st.write("No documents found.")

# Document Management Page
elif page == "Document Management":
    st.title("Document Management")
    st.markdown("Upload or delete documents.")
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
    if uploaded_file:
        if st.button("Upload"):
            with st.spinner("Uploading..."):
                doc_id = upload_document(uploaded_file)
                if doc_id:
                    st.success(f"Uploaded {uploaded_file.name} with ID: {doc_id}")
                    st.session_state.uploaded_docs.append({"filename": uploaded_file.name, "id": doc_id})
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
    st.metric("Corpus Size", f"{corpus_size:,} characters")
    st.metric("Total Tokens Used", st.session_state.query_stats['total_tokens'])
    if st.session_state.query_stats['times']:
        avg_time = sum(st.session_state.query_stats['times']) / len(st.session_state.query_stats['times'])
        st.metric("Average Response Time", f"{avg_time:.2f} seconds")
    else:
        st.metric("Average Response Time", "No queries yet")
    try:
        feedback_response = requests.get(f"{st.session_state.api_url}/feedback_stats", timeout=10)
        feedback_response.raise_for_status()
        feedback_stats = feedback_response.json()
        st.metric("Average Feedback Rating", f"{feedback_stats.get('avg_rating', 0):.2f}/5")
        st.metric("Positive Feedback Ratio", f"{feedback_stats.get('positive_ratio', 0):.2%}")
    except Exception as e:
        st.error(f"Failed to fetch feedback stats: {str(e)}")
        logger.error(f"Failed to fetch feedback stats: {str(e)}")
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
                    st.write(f"Tokens: {response.get('tokens_used', 0)}, Time: {response.get('response_time', 0):.2f}s")
    else:
        st.write("No queries yet.")