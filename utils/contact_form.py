import streamlit as st
from datetime import datetime
import os
import requests

def render_recommendation_with_contact(metadata: dict, key_suffix: str = ""):
    """Render recommendation message with contact form option"""
    
    query_id = metadata.get("query_id", "")
    checkbox_key = f"contact_checkbox_{query_id}_{key_suffix}"
    reset_key = f"reset_{checkbox_key}"
    success_key = f"contact_success_{query_id}_{key_suffix}"
    
    # CRITICAL: Handle reset flag BEFORE rendering the checkbox
    if st.session_state.get(reset_key):
        st.session_state[checkbox_key] = False
        st.session_state[reset_key] = False
        # Clear the reset flag and let natural rerun happen
    
    # Initialize checkbox state if not present
    if checkbox_key not in st.session_state:
        st.session_state[checkbox_key] = False
    
    st.markdown(
        "💡 **Recommendation:** I recommend our expert address this with more personalized and relevant detail. "
        "Would you like me to have a colleague reach out to you?"
    )
    
    # Checkbox for contact form
    show_form = st.checkbox(
        "Yes, have an expert contact me", 
        key=checkbox_key
    )
    
    # Show success feedback when form is dismissed after successful submission
    if not show_form and st.session_state.get(success_key):
        st.success("✅ Thank you! An expert will contact you within 24 hours.")
        st.balloons()
        # Clear success flag after showing it
        del st.session_state[success_key]
    
    # Show contact form if checked
    if show_form:
        show_contact_form_inline(metadata, key_suffix)

def show_contact_form_inline(metadata: dict, key_suffix: str = ""):
    """Display inline contact form"""
    
    query_id = metadata.get("query_id", "")
    checkbox_key = f"contact_checkbox_{query_id}_{key_suffix}"
    reset_key = f"reset_{checkbox_key}"
    success_key = f"contact_success_{query_id}_{key_suffix}"
    
    # FIX 1: Use metadata for the correct query and response
    original_query = metadata.get("query", "")
    if not original_query:
        original_query = metadata.get("question", "Question not available")
    original_response = metadata.get("answer", "")
    
    # Form styling
    st.markdown("""
        <style>
        div[data-testid="stForm"] button[kind="primary"] {
            background: linear-gradient(135deg, #105FAC 0%, #0d4a8a 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            font-weight: 600 !important;
        }
        div[data-testid="stForm"] button[kind="primary"]:hover {
            background: linear-gradient(135deg, #0d4a8a 0%, #105FAC 100%) !important;
            color: #FFFFFF !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(16, 95, 172, 0.3) !important;
        }
        div[data-testid="stForm"] button[kind="secondary"] {
            background-color: transparent !important;
            color: #575757 !important;
            border: 2px solid #D3D3D3 !important;
        }
        div[data-testid="stForm"] button[kind="secondary"]:hover {
            background-color: #f0f0f0 !important;
            border-color: #575757 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📞 Request Expert Consultation")
    st.caption("An expert will contact you")
    
    # Show the CORRECT question prominently
    if original_query and original_query != "Question not available":
        st.markdown("**Regarding your question:**")
        st.info(original_query)
    
    with st.form(key=f"contact_form_{query_id}_{key_suffix}", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name *", placeholder="John Doe")
            email = st.text_input("Email *", placeholder="john@example.com")
        
        with col2:
            phone = st.text_input("Phone Number", placeholder="+1 (555) 123-4567")
            company = st.text_input("Company", placeholder="Your Company")
        
        preferred_time = st.selectbox(
            "Preferred Contact Time",
            ["Morning (9 AM - 12 PM)", "Afternoon (12 PM - 5 PM)", "Evening (5 PM - 8 PM)", "Anytime"]
        )
        
        additional_notes = st.text_area(
            "Additional topics or questions you'd like to discuss",
            placeholder="e.g., Timeline requirements, specific concerns, budget considerations...",
            height=100
        )
        
        st.markdown("")
        
        col_submit, col_cancel = st.columns([1, 1])
        
        with col_submit:
            submitted = st.form_submit_button("📤 Submit Request", type="primary", use_container_width=True)
        
        with col_cancel:
            cancelled = st.form_submit_button("✖ Cancel", type="secondary", use_container_width=True)
        
        # Handle form submission
        if submitted:
            if not name or not email:
                st.error("Please provide your name and email.")
            elif '@' not in email:
                st.error("Please provide a valid email address.")
            else:
                success = log_contact_request(
                    query_id=query_id,
                    name=name,
                    email=email,
                    phone=phone,
                    company=company,
                    preferred_time=preferred_time,
                    additional_notes=additional_notes,
                    original_query=original_query,
                    original_response=original_response
                )
                if success:
                    # FIX 2: Set flags for success feedback and form dismissal
                    st.session_state[success_key] = True
                    st.session_state[reset_key] = True
                    # Form submit automatically triggers rerun
                else:
                    st.error("❌ Failed to submit request. Please try again.")
        
        # Handle form cancellation
        if cancelled:
            # FIX 2: Set reset flag to dismiss form on next run
            st.session_state[reset_key] = True
            # Form submit automatically triggers rerun

def log_contact_request(
    query_id: str,
    name: str,
    email: str,
    phone: str,
    company: str,
    preferred_time: str,
    additional_notes: str,
    original_query: str,
    original_response: str
) -> bool:
    """Submit contact request to backend API"""
    
    try:
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        
        auth_headers = {}
        if "access_token" in st.session_state:
            auth_headers["Authorization"] = f"Bearer {st.session_state.get('access_token')}"
        
        payload = {
            "query_id": query_id,
            "name": name,
            "email": email,
            "phone": phone or "",
            "company": company or "",
            "preferred_time": preferred_time,
            "additional_notes": additional_notes or "",
            "original_query": original_query,
            "original_response": original_response[:500] if original_response else ""
        }
        
        response = requests.post(
            f"{backend_url}/contact-requests",
            json=payload,
            headers=auth_headers,
            timeout=10
        )
        
        if response.status_code == 200:
            return True
        else:
            print(f"❌ Failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
