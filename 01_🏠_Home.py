import streamlit as st
from login import check_authentication, login_page, logout

print("======set page config======")

st.set_page_config(
    page_title="LCR Capital Partners",
    page_icon="🏢",
    layout="wide"
)
print("======end set page config======")

print("======Start imports======")
import requests
import json
import os
import time
from typing import Dict, Any, List, Optional
import re
from utils.styling import load_css, add_lcr_logo

from utils.contact_form import render_recommendation_with_contact


print("======End imports======")

import google.auth.transport.requests
import google.oauth2.id_token


def get_auth_headers(jwt_token=None):
    """Get authentication headers with JWT token"""
    headers = {}
    
    # Add App JWT token if provided
    if jwt_token and jwt_token != "":
        headers["Authorization"] = f"Bearer {jwt_token}"
    
    return headers

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

from utils.styling import apply_branding, add_footer, add_header


# Then in your main code, after apply_branding():
apply_branding()
add_header()


# And at the end of your page:
add_footer()


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

def get_admin_selected_model() -> str:
    """Get the model selected by admin"""
    try:
        response = requests.get(f"{BACKEND_URL}/admin/selected-model", headers=get_auth_headers(st.session_state.get("access_token")), timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("model", "gemini-pro")
    except:
        pass
    return "gemini-pro"

def check_backend_health() -> bool:
    """Check if backend is accessible"""
    try:
        # Health check doesn't need authentication
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def render_quality_metrics(metadata: Dict[str, Any], key_suffix: str = ""):
    """Render quality metrics with detected language info"""
    if not metadata:
        return
    
    # Extract quality data
    quality_data = metadata.get("quality", {})
    sources = metadata.get("sources", [])
    response_time = metadata.get("latency_ms", 0)
    query_id = metadata.get("query_id", "")
    detected_language = metadata.get("detected_language", "")  # NEW
    
    print(f"Query id in render quality metrics :{query_id}")
    # Only show expander if we have meaningful data
    if quality_data or sources or response_time:
        with st.expander("📊 Response Quality & Feedback", expanded=False):
            
            # Show detected language
            if detected_language:
                st.info(f"🌍 **Detected Language:** {detected_language}")
            
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
                    sources_count = metadata.get("sources_count", len(sources))
                    st.metric("Sources", sources_count)
            
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
            headers=get_auth_headers(st.session_state.get("access_token")),
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

def safe_render_content(content: str) -> str:
    """
    Safely prepare content for Streamlit markdown rendering
    """
    if not content:
        return ""
    
    # Only escape standalone dollar signs followed by digits
    # Don't touch already escaped ones or LaTeX blocks
    if '$$' not in content:  # Skip if it's a LaTeX block
        content = re.sub(r'(?<!\\)\$(\d)', r'\$\1', content)
    
    return content

def render_chat_interface():
    """Main chat interface"""
    
    # Dynamic header with session label
    session_label = st.session_state.get("session_label", "")
    if session_label:
        st.header(f"💬 RAG Chat Interface - {session_label}")
    else:
        st.header("💬 RAG Chat Interface")
    
    if not st.session_state.backend_connected:
        st.error("❌ Backend is not accessible. Please check your FastAPI server.")
        if st.button("🔄 Retry Connection"):
            st.session_state.backend_connected = check_backend_health()
            st.rerun()
        return
    
    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "metadata" in message:
                # Show recommendation with contact form option
                render_recommendation_with_contact(message["metadata"], key_suffix=f"_{i}")
                # Show quality metrics
                render_quality_metrics(message["metadata"], key_suffix=f"_{i}")


def render_chat_interface_old():
    """Main chat interface"""
    
    # Dynamic header with session label
    session_label = st.session_state.get("session_label", "")
    if session_label:
        st.header(f"💬 RAG Chat Interface - {session_label}")
    else:
        st.header("💬 RAG Chat Interface")
    
    if not st.session_state.backend_connected:
        st.error("❌ Backend is not accessible. Please check your FastAPI server.")
        if st.button("🔄 Retry Connection"):
            st.session_state.backend_connected = check_backend_health()
            st.rerun()
        return
    
    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "metadata" in message:
                st.markdown("💡 **Recommendation:** I recommend our expert address this with more personalized and relevant detail. Would you like me to have a colleague reach out to you?")
                render_quality_metrics(message["metadata"], key_suffix=f"_{i}")


def handle_user_input():
    """Handle user input - with safe rendering"""
    if prompt := st.chat_input("Ask about your documents...", key="main_chat_input"):
        
        # Add user message to history
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt, unsafe_allow_html=False)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("🚀🎇🎆 Generating response...🙂🤖"):
                response_data = call_backend_chat(prompt, st.session_state.selected_model)
                
                if response_data and "answer" in response_data:
                    # Extract and safely render answer
                    answer = response_data.get("answer", "")
                    safe_answer = safe_render_content(answer)
                    
                    # Display the response
                    st.markdown(safe_answer, unsafe_allow_html=False)

                    st.info("💡 **Recommendation:** I recommend our expert address this with more personalized and relevant detail. Would you like me to have a colleague reach out to you?")                    
                    # Add to history (store original, not escaped version)
                    assistant_message = {
                        "role": "assistant",
                        "content": answer,  # Store original
                        "metadata": response_data
                    }
                    st.session_state.messages.append(assistant_message)
                    
                    # Display quality metrics
                    render_quality_metrics(response_data, key_suffix="current")
                    
                else:
                    error_msg = "❌ Sorry, I couldn't generate a response. Please try again."
                    st.error(error_msg)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "metadata": {}
                    })
        
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
                headers=get_auth_headers(st.session_state.get("access_token")),
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
        response = requests.get(f"{BACKEND_URL}/documents", headers=get_auth_headers(st.session_state.get("access_token")), timeout=10)
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
            headers=get_auth_headers(st.session_state.get("access_token")),
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


def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = get_admin_selected_model()
    
    if "backend_connected" not in st.session_state:
        st.session_state.backend_connected = check_backend_health()
    
    # Add session ID for conversation persistence
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = None

def call_backend_chat(query: str, model: str) -> Optional[Dict[str, Any]]:
    """Call FastAPI backend for chat response with session persistence"""
    try:
        payload = {
            "query": query,
            "model": model,
            "session_id": st.session_state.chat_session_id,
            "session_label": st.session_state.get("session_label", "")
        }

        response = requests.post(
            f"{BACKEND_URL}/chat",
            json=payload,
            headers=get_auth_headers(st.session_state.get("access_token")),
            timeout=90,
        )

        if response.status_code == 200:
            data = response.json()
            
            # 🎨 FRONTEND ENCODING CHECK - Response received
            print(f"\n{'='*60}")
            print(f"🎨 FRONTEND - Response received from backend")
            print(f"  Status: {response.status_code}")
            print(f"  Answer length: {len(data.get('answer', ''))} chars")
            
            answer = data.get('answer', '')
            if 'I-5' in answer:
                i5_start = answer.find('I-5')
                i5_section = answer[i5_start:i5_start+30]
                print(f"  ⚠️ I-5 in frontend answer: '{i5_section}'")
                print(f"  🔬 Hex: {i5_section.encode('unicode-escape').decode('ascii')}")
                
                if 'Ƒѵ' in i5_section or 'ƒ' in i5_section:
                    print(f"  ❌ CORRUPTION DETECTED in frontend!")
            
            print(f"  Answer preview: {answer[:200]}...")
            print(f"{'='*60}\n")

            if "session_id" in data and not st.session_state.chat_session_id:
                st.session_state.chat_session_id = data["session_id"]
            # ✅ ADD QUERY TO METADATA - Ensure original question is available
            data['query'] = query  # Add the original query to response data
            
            return data
        else:
            st.error(f"Backend error ({response.status_code}): {response.text}")
            return None

    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Connection error. Please check backend status.")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        print(f"Frontend error details: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def render_sidebar():
    """Compact sidebar with icons and metrics"""
    with st.sidebar:
        # User info at top
        username = st.session_state.get('username', 'User')
        user_role = st.session_state.get('user_role', 'user')
        
        # Navigation buttons
                
        # Show system stats only for admin/superuser
        if user_role in ["admin", "superuser"]:
            status_icon = "🟢" if st.session_state.backend_connected else "🔴"
            st.markdown(f"### {status_icon} System")
            
            try:
                response = requests.get(f"{BACKEND_URL}/admin/system-stats", headers=get_auth_headers(st.session_state.get("access_token")), timeout=5)
                stats = response.json() if response.status_code == 200 else {}
            except:
                stats = {}
            
            st.markdown("---")
            
            st.markdown(f"🤖 **Model**")
            st.caption(st.session_state.selected_model)
            
            st.markdown(f"👥 **Sessions**")
            st.caption(f"{stats.get('active_sessions', 0)} active / {stats.get('total_sessions', 0)} total")
            
            st.markdown(f"⚡ **Avg Response**")
            st.caption(f"{stats.get('avg_response_time', 0):.1f}s")
            
            st.markdown(f"📄 **Top Document**")
            top_doc = stats.get('top_document', 'None')
            doc_count = stats.get('top_document_count', 0)
            if top_doc != 'None':
                st.caption(f"{top_doc}")
                st.caption(f"↳ {doc_count} references")
            else:
                st.caption("No data yet")
        else:
            # Minimal info for regular users
            st.markdown("### 💬 Chat")
            if st.session_state.chat_session_id:
                st.caption(f"{len(st.session_state.messages)} messages")
            else:
                st.caption("Start a conversation")

def load_previous_sessions():
    """Load list of previous sessions from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/sessions", headers=get_auth_headers(st.session_state.get("access_token")), timeout=10)
        if response.status_code == 200:
            st.session_state.available_sessions = response.json()
            st.success(f"Found {len(st.session_state.available_sessions)} sessions")
        else:
            st.error("Failed to load sessions")
    except Exception as e:
        st.error(f"Error loading sessions: {str(e)}")

def restore_session(session_id: str):
    """Restore a previous chat session"""
    try:
        # Get session details first
        session_response = requests.get(f"{BACKEND_URL}/sessions", headers=get_auth_headers(st.session_state.get("access_token")), timeout=10)
        if session_response.status_code == 200:
            sessions = session_response.json()
            current_session = next((s for s in sessions if s["session_id"] == session_id), None)
            
            if current_session:
                st.session_state.session_label = current_session.get("session_label", "")
        
        # Get messages
        response = requests.get(f"{BACKEND_URL}/sessions/{session_id}/messages", headers=get_auth_headers(st.session_state.get("access_token")), timeout=10)
        if response.status_code == 200:
            messages_data = response.json()
            
            st.session_state.messages = []
            
            for msg in messages_data:
                st.session_state.messages.append({
                    "role": "user",
                    "content": msg["query"]
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": msg["response"],
                    "metadata": {"quality": {"score": msg.get("quality_score", 0)}}
                })
            
            st.session_state.chat_session_id = session_id
            st.success(f"Restored {len(messages_data)} exchanges")
            st.rerun()
        else:
            st.error("Failed to restore session")
    except Exception as e:
        st.error(f"Error restoring session: {str(e)}")

def render_settings_tab():
    """Settings and session management"""
    st.subheader("⚙️ System Settings")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 💬 Current Session")
        
        if st.session_state.chat_session_id:
            # Show current session
            current_label = st.session_state.get("session_label", "")
            st.info(f"Session: `{st.session_state.chat_session_id[:8]}...`")
            
            # Allow labeling
            new_label = st.text_input(
                "Session Label (e.g., customer name):",
                value=current_label,
                placeholder="Enter label or leave blank for timestamp"
            )
            
            if st.button("💾 Save Label") and new_label != current_label:
                save_session_label(st.session_state.chat_session_id, new_label)
                st.session_state.session_label = new_label
                st.success("Label saved!")
            
            st.caption(f"{len(st.session_state.messages)} messages in conversation")
        else:
            st.info("No active session")
        
        st.markdown("---")
        
        # Session actions
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🆕 New Session"):
                st.session_state.messages = []
                st.session_state.chat_session_id = None
                st.session_state.session_label = ""
                st.rerun()
        
        with col_b:
            if st.button("📋 Load Session"):
                load_previous_sessions()
        
        # Session selector
        if "available_sessions" in st.session_state and st.session_state.available_sessions:
            selected = st.selectbox(
                "Select session to restore:",
                options=st.session_state.available_sessions,
                format_func=lambda s: format_session_display(s)
            )
            
            if st.button("🔄 Restore Selected"):
                restore_session(selected["session_id"])    
    with col2:
        st.markdown("### 📊 Recent Sessions (Last 10)")
        
        if st.button("🔄 Load Summaries"):
            load_session_summaries()
        
        # Show session summaries
        if "session_summaries" in st.session_state and st.session_state.session_summaries:
            for idx, summary in enumerate(st.session_state.session_summaries, 1):
                with st.expander(
                    f"#{idx} {summary['display_name']} ({summary['message_count']} msgs)",
                    expanded=False
                ):
                    st.caption(f"**Created:** {summary['created_at']}")
                    st.caption(f"**Last Active:** {summary['last_activity']}")
                    st.markdown("**Conversation:**")
                    st.text(summary['summary'])
        else:
            st.info("Click 'Load Summaries' to view recent sessions")

def format_session_display(session):
    """Format session for display in selector"""
    label = session.get("session_label", "")
    created = session.get("created_at", "")
    msg_count = session.get("message_count", 0)
    
    if label:
        return f"{label} ({msg_count} msgs)"
    else:
        # Use timestamp as fallback
        from datetime import datetime
        dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
        return f"{dt.strftime('%Y-%m-%d %H:%M')} ({msg_count} msgs)"

def save_session_label(session_id: str, label: str):
    """Save session label to backend"""
    try:
        response = requests.patch(
            f"{BACKEND_URL}/sessions/{session_id}/label",
            json={"label": label},
            headers=get_auth_headers(st.session_state.get("access_token")),
            timeout=10
        )
        return response.status_code == 200
    except:
        return False

def load_previous_sessions():
    """Load list of previous sessions"""
    try:
        response = requests.get(f"{BACKEND_URL}/sessions", headers=get_auth_headers(st.session_state.get("access_token")), timeout=10)
        if response.status_code == 200:
            st.session_state.available_sessions = response.json()
            st.success(f"Found {len(st.session_state.available_sessions)} sessions")
    except Exception as e:
        st.error(f"Error loading sessions: {str(e)}")

def load_session_summaries():
    """Load session summaries for last 10 sessions"""
    try:
        response = requests.get(f"{BACKEND_URL}/sessions/summaries", headers=get_auth_headers(st.session_state.get("access_token")), timeout=10)
        if response.status_code == 200:
            st.session_state.session_summaries = response.json()
    except Exception as e:
        st.error(f"Error loading summaries: {str(e)}")


def main_old():
    """Main application entry point"""
    print("Starting....")
    if not check_authentication():
        login_page()
        return
    load_styles_safely()
    print("======end load styles safely======")
    initialize_session_state()
    print("======end session init======")
    render_sidebar()
    print("======end sidebar render======")
    
    # Add Settings tab
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Documents", "⚙️ Settings"])
    
    with tab1:
        render_chat_interface()
        handle_user_input()
    
    with tab2:
        render_document_management()
    
    with tab3:
        render_settings_tab()


def render_document_list_readonly():
    """Read-only document list for regular users"""
    st.subheader("📄 Available Documents")
    st.info("ℹ️ You can view available documents. Contact admin to upload new documents.")
    
    try:
        documents = get_document_list()
        
        if documents:
            st.markdown(f"**{len(documents)} documents available:**")
            for i, doc in enumerate(documents, 1):
                icon = "📄" if doc.endswith(('.txt', '.md')) else "📊" if doc.endswith(('.csv', '.json')) else "📑"
                st.write(f"{i}. {icon} {doc}")
        else:
            st.info("📭 No documents available yet.")
            
    except Exception as e:
        st.error(f"❌ Error loading documents: {str(e)}")

def render_navigation_sidebar():
    """Navigation sidebar for admin"""
    with st.sidebar:
        username = st.session_state.get('username', 'User')
        user_role = st.session_state.get('user_role', 'user')
        
        st.markdown(f"👤 **{username}**")
        st.caption(f"Role: {user_role}")
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            from login import logout
            logout()

def main():
    """Main application entry point"""
    
    # Check authentication first
    if not check_authentication():
        login_page()
        return
    
    render_navigation_sidebar()
    # Existing app code
    load_styles_safely()
    initialize_session_state()
    render_sidebar()
    
    # Get user role
    user_role = st.session_state.get("user_role", "user")
    
    # Show tabs based on role
    if user_role in ["admin", "superuser"]:
        # Full access for admin/superuser
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Documents", "⚙️ Settings"])
        
        with tab1:
            render_chat_interface()
            handle_user_input()
        
        with tab2:
            render_document_management()
        
        with tab3:
            render_settings_tab()
    else:
        # User only gets chat - documents are read-only view
        tab1, tab2 = st.tabs(["💬 Chat", "📄 Documents"])
        
        with tab1:
            render_chat_interface()
            handle_user_input()
        
        with tab2:
            render_document_list_readonly()


if __name__ == "__main__":
    main()
