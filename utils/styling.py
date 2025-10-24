import streamlit as st
from pathlib import Path
import base64

def load_css():
    """Load custom CSS with error handling"""
    css_path = Path("assets/styles.css")
    if css_path.exists():
        try:
            with open(css_path, encoding='utf-8') as f:
                css_content = f.read()
                st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
                print("✓ CSS loaded successfully")
        except Exception as e:
            print(f"✗ Error loading CSS: {e}")
    else:
        print(f"✗ CSS file not found at: {css_path}")

def add_lcr_logo():
    """Add LCR logo to sidebar with proper sizing to avoid clipping"""
    logo_path = Path("assets/lcr_logo.png")
    
    if logo_path.exists():
        try:
            with open(logo_path, "rb") as f:
                logo_base64 = base64.b64encode(f.read()).decode()
            
            # Optimized logo with increased height to prevent clipping
            st.markdown(f"""
            <style>
            /* Logo in sidebar - prevent clipping */
            section[data-testid="stSidebar"] > div:first-child::before {{
                content: '';
                display: block;
                width: 100%;
                height: 180px;
                background-image: url('data:image/png;base64,{logo_base64}');
                background-repeat: no-repeat;
                background-position: center center;
                background-size: contain;
                margin: 0.5rem 0 0.25rem 0;
                padding: 0.5rem;
            }}
            </style>
            """, unsafe_allow_html=True)
            print("✓ Logo loaded successfully")
        except Exception as e:
            print(f"✗ Error loading logo: {e}")
    else:
        print(f"✗ Logo file not found at: {logo_path}")
    
    # Single divider after logo with reduced spacing
    st.sidebar.markdown("<hr style='margin: 0.25rem 0 0.5rem 0; border-color: rgba(255, 255, 255, 0.3);'>", unsafe_allow_html=True)

def apply_branding():
    """Apply all branding elements"""
    print("Applying branding...")
    load_css()
    add_lcr_logo()
    print("Branding applied")

def add_header():
    """Add branded header bar - above everything"""
    st.markdown("""
    <div style='
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 70px;
        background: linear-gradient(135deg, #105FAC 0%, #397B5C 100%);
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        z-index: 999999;
    '>
        <div style='
            font-family: "Playfair Display", Georgia, serif;
            color: white;
            font-size: 1.8rem;
            font-weight: 700;
            letter-spacing: 0.05em;
        '>
            LCR Capital Partners
        </div>
        <div style='
            font-family: "Lato", Arial, sans-serif;
            color: #87AFD5;
            font-size: 0.95rem;
            font-style: italic;
        '>
            Transforming lives through opportunities
        </div>
    </div>
    """, unsafe_allow_html=True)

def add_footer():
    """Add branded footer with proper z-index"""
    st.markdown("""
    <div style='
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        height: 60px;
        background: linear-gradient(135deg, #105FAC 0%, #397B5C 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 2rem;
        box-shadow: 0 -4px 15px rgba(0,0,0,0.3);
        z-index: 999999;
        color: white;
        font-family: "Lato", Arial, sans-serif;
        font-size: 0.9rem;
    '>
        © 2025 LCR Capital Partners. All rights reserved. | 
        <a href='https://www.lcrcapital.com' style='
            color: #DD8C1C;
            text-decoration: none;
            margin-left: 0.5rem;
            font-weight: 600;
        '>
            www.lcrcapital.com
        </a>
    </div>
    """, unsafe_allow_html=True)
