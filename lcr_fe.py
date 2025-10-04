import streamlit as st
import requests
import json
import os
import time
from typing import Dict, Any, List, Optional

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG Chat System",
    page_icon="🧠",
    layout="wide"
)

def load_styles_safely():
    """Minimal CSS - no conflicts"""
    st.markdown("""
    <style>
    .stApp { 
        background: #0E1117; 
        color: #E6E9EF; 
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "selected_model" not in st.session_state:
        # Try to get model from admin selection
        st.session_state.selected_model = get_admin_selected_model()
    
    if "backend_connected" not in st.session_state:
        st.session_state.backend_connected = check_backend_health()

def get_admin_selected_model() -> str:
    """Get the model selected by admin"""
    try:
        response = requests.get(f"{BACKEND_URL}/admin/selected-model", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("model", "gemini-pro")
    except:
        pass
    return "gemini-pro"

def check_backend_health() -> bool:
    """Check if backend is accessible"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def call_backend_chat(query: str, model: str) -> Optional[Dict[str, Any]]:
    """Call FastAPI backend for chat response"""
    try:
        payload = {"query": query, "model": model}
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json=payload,
            timeout=90,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Backend error ({response.status_code}): {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Cannot connect to backend. Please check if FastAPI server is running.")
        st.session_state.backend_connected = False
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

def render_quality_metrics(metadata: Dict[str, Any], key_suffix: str = ""):
    """Render quality metrics with feedback collection"""
    if not metadata:
        return
    
    # Extract quality data
    quality_data = metadata.get("quality", {})
    sources = metadata.get("sources", [])
    response_time = metadata.get("latency_ms", 0)
    query_id = metadata.get("query_id", "")
    
    # Only show expander if we have meaningful data
    if quality_data or sources or response_time:
        with st.expander("📊 Response Quality & Feedback", expanded=False):
            
            # Metrics row
            if quality_data:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    score = quality_data.get("score", 0)
                    st.metric("Quality Score", f"{score:.3f}")
                
                with col2:
                    relevance = quality_data.get("retrieval_relevance", 0)
                    st.metric("Relevance", f"{relevance:.3f}")
                
                with col3:
                    coverage = quality_data.get("coverage", 0)
                    st.metric("Coverage", f"{coverage:.3f}")
                
                with col4:
                    sources_used = quality_data.get("sources_used", 0)
                    st.metric("Sources", sources_used)
            
            # Response time
            if response_time > 0:
                st.caption(f"⏱️ Response generated in {response_time:.0f}ms")
            
            # Sources information
            if sources:
                st.markdown("**📚 Retrieved Sources:**")
                for idx, source in enumerate(sources[:5]):
                    if isinstance(source, dict):
                        source_name = source.get("source", f"Source {idx+1}")
                        similarity = source.get("similarity", 0)
                        st.write(f"• **{source_name}** (similarity: {similarity:.3f})")
            
            st.divider()
            
            # FEEDBACK COLLECTION
            st.markdown("**💬 Rate This Response:**")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Thumbs up/down
                feedback_col1, feedback_col2 = st.columns(2)
                
                with feedback_col1:
                    if st.button("👍 Helpful", key=f"thumbs_up_{key_suffix}", use_container_width=True):
                        submit_feedback(query_id, metadata, thumbs_up=True, rating=5)
                        st.success("Thanks for your feedback!")
                
                with feedback_col2:
                    if st.button("👎 Not Helpful", key=f"thumbs_down_{key_suffix}", use_container_width=True):
                        submit_feedback(query_id, metadata, thumbs_up=False, rating=1)
                        st.warning("Feedback recorded. We'll improve!")
            
            with col2:
                # Star rating
                rating = st.select_slider(
                    "Detailed Rating:",
                    options=[1, 2, 3, 4, 5],
                    value=3,
                    key=f"rating_{key_suffix}",
                    format_func=lambda x: "⭐" * x
                )
            
            # Optional comment
            feedback_comment = st.text_area(
                "Additional Comments (optional):",
                key=f"comment_{key_suffix}",
                placeholder="What could be improved?",
                height=80
            )
            
            # Submit detailed feedback
            if st.button("📤 Submit Detailed Feedback", key=f"submit_{key_suffix}"):
                submit_feedback(
                    query_id, 
                    metadata, 
                    rating=rating, 
                    comment=feedback_comment,
                    thumbs_up=(rating >= 3)
                )
                st.success("✅ Detailed feedback submitted!")

def submit_feedback(
    query_id: str, 
    metadata: Dict[str, Any], 
    thumbs_up: bool = None,
    rating: int = None,
    comment: str = ""
):
    """Submit feedback to backend"""
    try:
        # Extract data from metadata
        question = st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else ""
        response = st.session_state.messages[-1]["content"] if st.session_state.messages else ""
        
        payload = {
            "query_id": query_id,
            "question": question,
            "response": response,
            "thumbs_up": thumbs_up,
            "rating": rating,
            "feedback_text": comment,
            "quality_scores": metadata.get("quality", {}),
            "sources": metadata.get("sources", []),
            "response_time": metadata.get("latency_ms", 0),
            "model_used": metadata.get("model_used", "unknown")
        }
        
        response = requests.post(
            f"{BACKEND_URL}/feedback",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return True
        else:
            st.error(f"Failed to submit feedback: {response.status_code}")
            return False
            
    except Exception as e:
        st.error(f"Error submitting feedback: {str(e)}")
        return False


def render_chat_interface():
    """Main chat interface with fixed rendering"""
    st.header("💬 RAG Chat Interface")
    
    if not st.session_state.backend_connected:
        st.error("❌ Backend is not accessible. Please check your FastAPI server.")
        if st.button("🔄 Retry Connection"):
            st.session_state.backend_connected = check_backend_health()
            st.rerun()
        return
    
    # Display chat history with proper markdown rendering
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Force proper markdown rendering
            st.markdown(message["content"], unsafe_allow_html=False)
            
            # Render quality metrics for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                render_quality_metrics(message["metadata"], key_suffix=f"_{i}")

def handle_user_input():
    """Handle user input and generate responses - FIXED VERSION"""
    if prompt := st.chat_input("Ask about your documents...", key="main_chat_input"):
        
        # Add user message to history
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt, unsafe_allow_html=False)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("🤔 Generating response..."):
                response_data = call_backend_chat(prompt, st.session_state.selected_model)
                
                if response_data and "answer" in response_data:
                    # Extract clean answer text
                    answer = response_data["answer"]
                    
                    # Display the response with proper markdown
                    message_placeholder.markdown(answer, unsafe_allow_html=False)
                    
                    # Add assistant message to history
                    assistant_message = {
                        "role": "assistant",
                        "content": answer,
                        "metadata": response_data
                    }
                    st.session_state.messages.append(assistant_message)
                    
                    # Display quality metrics below
                    render_quality_metrics(response_data, key_suffix="current")
                    
                else:
                    error_msg = "❌ Sorry, I couldn't generate a response. Please try again."
                    message_placeholder.error(error_msg)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "metadata": {}
                    })
        
        # Force rerun to update display
        time.sleep(0.1)
        st.rerun()

def render_document_management():
    """Document management interface"""
    st.header("📄 Document Management")
    
    if not st.session_state.backend_connected:
        st.error("❌ Backend connection required for document management.")
        return
    
    # Upload section
    st.subheader("📤 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['txt', 'pdf', 'docx', 'md', 'json', 'csv'],
        accept_multiple_files=True,
        help="Supported formats: TXT, PDF, DOCX, MD, JSON, CSV"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("📤 Upload & Process Documents", type="primary", disabled=not uploaded_files):
            if uploaded_files:
                upload_documents(uploaded_files)
    
    with col2:
        st.metric("Selected Files", len(uploaded_files) if uploaded_files else 0)
    
    st.divider()
    
    # Document listing and management
    st.subheader("📋 Current Documents")
    
    try:
        documents = get_document_list()
        
        if documents:
            # Document selection for management
            selected_docs = st.multiselect(
                "Select documents to manage:",
                options=documents,
                help="Select documents for batch operations"
            )
            
            # Management actions
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                if st.button("🗑️ Delete Selected", 
                           type="secondary", 
                           disabled=not selected_docs):
                    if selected_docs:
                        delete_documents(selected_docs)
            
            with col2:
                if st.button("🔄 Refresh"):
                    st.rerun()
            
            with col3:
                st.metric("Selected", len(selected_docs))
            
            with col4:
                st.metric("Total", len(documents))
            
            # Display document list
            st.markdown("**Available Documents:**")
            for i, doc in enumerate(documents, 1):
                icon = "📄" if doc.endswith(('.txt', '.md')) else "📊" if doc.endswith(('.csv', '.json')) else "📑"
                st.write(f"{i}. {icon} {doc}")
                
        else:
            st.info("📭 No documents found. Upload some documents to get started!")
            
    except Exception as e:
        st.error(f"❌ Error loading documents: {str(e)}")

def upload_documents(uploaded_files) -> None:
    """Handle document upload to backend"""
    try:
        files = []
        for uploaded_file in uploaded_files:
            file_content = uploaded_file.getvalue()
            files.append(("files", (uploaded_file.name, file_content, uploaded_file.type)))
        
        with st.spinner(f"📤 Uploading {len(uploaded_files)} documents..."):
            response = requests.post(
                f"{BACKEND_URL}/upload",
                files=files,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                uploaded_count = result.get("uploaded_count", len(uploaded_files))
                st.success(f"✅ Successfully uploaded {uploaded_count} documents!")
                st.rerun()
            else:
                st.error(f"❌ Upload failed: {response.text}")
                
    except Exception as e:
        st.error(f"Upload error: {str(e)}")

def get_document_list() -> List[str]:
    """Retrieve list of documents from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get("documents", data.get("sources", []))
        return []
        
    except Exception as e:
        st.error(f"❌ Error fetching documents: {str(e)}")
        return []

def delete_documents(document_names: List[str]) -> None:
    """Delete selected documents from backend"""
    try:
        response = requests.delete(
            f"{BACKEND_URL}/documents",
            json={"documents": document_names},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            deleted_count = result.get("deleted_count", len(document_names))
            st.success(f"🗑️ Successfully deleted {deleted_count} documents!")
            st.rerun()
        else:
            st.error(f"❌ Deletion failed: {response.text}")
            
    except Exception as e:
        st.error(f"Delete error: {str(e)}")

def render_sidebar():
    """Enhanced sidebar with system status"""
    with st.sidebar:
        st.title("⚙️ System Control")
        
        # Backend status
        status_color = "🟢" if st.session_state.backend_connected else "🔴"
        status_text = "Connected" if st.session_state.backend_connected else "Disconnected"
        st.markdown(f"**Backend:** {status_color} {status_text}")
        
        # Current model display
        st.markdown(f"**Active Model:** `{st.session_state.selected_model}`")
        st.caption("Model is set via Admin Panel")
        
        # System controls
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh"):
                st.session_state.backend_connected = check_backend_health()
                st.session_state.selected_model = get_admin_selected_model()
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        
        # Chat statistics
        if st.session_state.messages:
            user_msgs = sum(1 for msg in st.session_state.messages if msg["role"] == "user")
            assistant_msgs = sum(1 for msg in st.session_state.messages if msg["role"] == "assistant")
            
            st.divider()
            st.markdown("**Chat Statistics:**")
            st.metric("Questions", user_msgs)
            st.metric("Responses", assistant_msgs)

def main():
    """Main application entry point"""
    load_styles_safely()
    initialize_session_state()
    render_sidebar()
    
    # Main content tabs
    tab1, tab2 = st.tabs(["💬 Chat", "📄 Documents"])
    
    with tab1:
        render_chat_interface()
        handle_user_input()
    
    with tab2:
        render_document_management()

if __name__ == "__main__":
    main()
