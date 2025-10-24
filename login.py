import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

import google.auth.transport.requests
import google.oauth2.id_token



def check_authentication():
    """Check if user is authenticated"""
    return "access_token" in st.session_state and st.session_state.access_token

def login_page():
    """Render login page"""
    st.title("🔐 RAG System Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Sign In")
        
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🔓 Login", type="primary", use_container_width=True):
                if username and password:
                    login_user(username, password)
                else:
                    st.error("Please enter username and password")
        
        with col_b:
            if st.button("📝 Register", use_container_width=True):
                st.session_state.show_register = False
                st.rerun()
        
        # Show registration form if requested
        if st.session_state.get("show_register", False):
            st.markdown("---")
            st.markdown("### Create Account")
            
            new_username = st.text_input("Username", key="reg_username")
            new_email = st.text_input("Email", key="reg_email")
            new_password = st.text_input("Password", type="password", key="reg_password")
            new_full_name = st.text_input("Full Name (optional)", key="reg_fullname")
            
            col_x, col_y = st.columns(2)
            with col_x:
                if st.button("✅ Create Account", type="primary"):
                    register_user(new_username, new_email, new_password, new_full_name)
            with col_y:
                if st.button("❌ Cancel"):
                    st.session_state.show_register = False
                    st.rerun()

def login_user(username: str, password: str):
    """Authenticate user"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/token",
            data={"username": username, "password": password},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            st.session_state.access_token = data["access_token"]
            st.session_state.user_role = data["role"]
            st.session_state.username = username
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid credentials")
    except Exception as e:
        st.error(f"Login error: {str(e)}")

def register_user(username: str, email: str, password: str, full_name: str):
    """Register new user"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/register",
            params={
                "username": username,
                "email": email,
                "password": password,
                "full_name": full_name
            },
            timeout=10
        )
        
        if response.status_code == 200:
            st.success("✅ Account created! Please login.")
            st.session_state.show_register = False
            st.rerun()
        else:
            st.error(f"❌ Registration failed: {response.json().get('detail', 'Unknown error')}")
    except Exception as e:
        st.error(f"Registration error: {str(e)}")

def logout():
    """Logout user"""
    for key in ["access_token", "user_role", "username"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()
