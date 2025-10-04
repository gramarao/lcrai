import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Quality Dashboard", page_icon="📊", layout="wide")

API_BASE_URL = "http://localhost:8000"

st.title("📊 Quality Analysis Dashboard")

# Test API connection
try:
    response = requests.get(f"{API_BASE_URL}/health", timeout=5)
    if response.status_code == 200:
        st.success("🟢 Connected to API")
    else:
        st.error("🔴 API connection failed")
        st.stop()
except:
    st.error("🔴 Cannot connect to API. Make sure it's running at http://localhost:8000")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Analytics", "📊 Feedback Stats", "⚙️ System Info"])

with tab1:
    st.header("System Analytics")
    
    try:
        # Get analytics from backend
        response = requests.get(f"{API_BASE_URL}/admin/analytics", timeout=10)
        
        if response.status_code == 200:
            analytics = response.json()
            system_status = analytics.get('system_status', {})
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Documents", system_status.get('documents', 0))
            
            with col2:
                st.metric("Chunks", system_status.get('chunks', 0))
            
            with col3:
                st.metric("Messages", system_status.get('messages', 0))
            
            with col4:
                st.metric("Avg Quality", f"{system_status.get('avg_quality', 0):.3f}")
            
            st.divider()
            
            # Response time
            st.metric("Avg Response Time", f"{system_status.get('avg_response_time_ms', 0):.0f} ms")
            
            # Model usage
            if analytics.get('model_usage'):
                st.subheader("Model Usage")
                model_df = pd.DataFrame(analytics['model_usage'])
                fig = px.bar(model_df, x='model', y='count', title='Queries by Model')
                st.plotly_chart(fig, use_container_width=True)
            
            # Feedback stats
            if analytics.get('feedback_stats'):
                st.subheader("Feedback Statistics")
                feedback_df = pd.DataFrame(analytics['feedback_stats'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(feedback_df, x='model', y='avg_rating', 
                                title='Average Rating by Model')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(feedback_df, x='model', y='count',
                                title='Feedback Count by Model')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not load analytics")
    
    except Exception as e:
        st.error(f"Error loading analytics: {e}")

with tab2:
    st.header("Feedback Analysis")
    
    try:
        response = requests.get(f"{API_BASE_URL}/feedback/analytics", timeout=10)
        
        if response.status_code == 200:
            analytics = response.json()
            summary = analytics.get('summary', {})
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Feedback", summary.get('total_feedback', 0))
            
            with col2:
                st.metric("Positive Rate", f"{summary.get('positive_rate', 0):.1f}%")
            
            with col3:
                st.metric("Avg Rating", f"{summary.get('avg_rating', 0):.2f}/5")
            
            with col4:
                st.metric("Avg Relevance", f"{summary.get('avg_relevance', 0):.3f}")
            
            st.divider()
            
            # Quality scores
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Avg Accuracy", f"{summary.get('avg_accuracy', 0):.3f}")
            
            with col2:
                st.metric("Avg Completeness", f"{summary.get('avg_completeness', 0):.3f}")
            
            # Rating distribution
            if analytics.get('rating_distribution'):
                st.subheader("Rating Distribution")
                rating_df = pd.DataFrame(analytics['rating_distribution'])
                
                fig = px.bar(rating_df, x='rating', y='count',
                           title='Distribution of Ratings',
                           labels={'rating': 'Rating (stars)', 'count': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent negative feedback
            if analytics.get('recent_negative_feedback'):
                st.subheader("Recent Issues (Low Ratings)")
                
                for fb in analytics['recent_negative_feedback'][:10]:
                    with st.expander(f"⭐ {fb.get('rating')}/5 - {fb.get('question', '')[:80]}..."):
                        st.write(f"**Comment:** {fb.get('comment', 'No comment')}")
                        st.caption(f"Query ID: {fb.get('query_id')}")
                        st.caption(f"Time: {fb.get('timestamp')}")
        else:
            st.warning("No feedback data available")
    
    except Exception as e:
        st.error(f"Error loading feedback: {e}")

with tab3:
    st.header("System Information")
    
    try:
        # Health check
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            health = response.json()
            
            st.success(f"✅ System Status: {health.get('status', 'unknown').upper()}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Documents", health.get('documents', 0))
            
            with col2:
                st.metric("Chunks", health.get('chunks', 0))
            
            with col3:
                st.metric("Embedding Dim", health.get('embedding_dimension', 0))
            
            st.divider()
            
            # Current configuration
            st.subheader("Current Configuration")
            
            # Get model info
            model_response = requests.get(f"{API_BASE_URL}/admin/selected-model", timeout=5)
            if model_response.status_code == 200:
                model_data = model_response.json()
                st.write(f"**Active Model:** `{model_data.get('model')}`")
                st.write(f"**Last Updated:** {model_data.get('last_updated')}")
            
            # Get feedback config
            fb_response = requests.get(f"{API_BASE_URL}/admin/feedback-config", timeout=5)
            if fb_response.status_code == 200:
                fb_data = fb_response.json()
                st.write(f"**Feedback Enhanced:** {'✅ Enabled' if fb_data.get('feedback_enabled') else '❌ Disabled'}")
                st.write(f"**Feedback Weight:** {fb_data.get('feedback_weight', 0):.2f}")
            
            st.caption(f"Last checked: {health.get('timestamp')}")
        else:
            st.error("System health check failed")
    
    except Exception as e:
        st.error(f"Error checking system health: {e}")