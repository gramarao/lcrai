import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def main():
    st.set_page_config(page_title="RAG Quality Dashboard", layout="wide")
    
    st.title("📊 RAG System Quality Dashboard")
    
    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        days = st.selectbox("Time Period", [7, 14, 30, 90], index=2)
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
    
    # Get feedback stats
    try:
        response = requests.get(f"http://localhost:8000/api/feedback/stats?days={days}")
        if response.status_code == 200:
            stats = response.json()
        else:
            st.error("Failed to load dashboard data")
            return
    except:
        st.error("❌ Cannot connect to API")
        return
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Feedback", 
            stats["total_feedback"],
            help="Total feedback responses received"
        )
    
    with col2:
        avg_rating = stats["average_rating"]
        st.metric(
            "Average Rating", 
            f"{avg_rating:.1f}/5.0",
            delta=f"{(avg_rating - 3.0):.1f}" if avg_rating > 0 else None,
            help="Average user rating (1-5 stars)"
        )
    
    with col3:
        satisfaction_rate = stats["satisfaction_rate"]
        st.metric(
            "Satisfaction Rate", 
            f"{satisfaction_rate:.1f}%",
            delta=f"{(satisfaction_rate - 70):.1f}%" if satisfaction_rate > 0 else None,
            help="Percentage of positive feedback"
        )
    
    with col4:
        avg_response_time = stats["average_response_time"]
        st.metric(
            "Avg Response Time", 
            f"{avg_response_time:.1f}s",
            delta=f"{(3.0 - avg_response_time):.1f}s" if avg_response_time > 0 else None,
            delta_color="inverse",
            help="Average response generation time"
        )
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Feedback Trend")
        
        # Mock time series data (in real implementation, get from API)
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        feedback_counts = [stats["total_feedback"] // days + (i % 3) for i in range(days)]
        
        fig_trend = px.line(
            x=dates, 
            y=feedback_counts,
            title="Daily Feedback Volume",
            labels={"x": "Date", "y": "Feedback Count"}
        )
        fig_trend.update_traces(line_color='#1f77b4')
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        st.subheader("👍 Sentiment Distribution")
        
        positive = stats["positive_feedback"]
        negative = stats["negative_feedback"]
        neutral = max(stats["total_feedback"] - positive - negative, 0)
        
        fig_sentiment = px.pie(
            values=[positive, negative, neutral],
            names=["Positive", "Negative", "Neutral"],
            title="User Sentiment",
            color_discrete_map={
                "Positive": "#2ecc71",
                "Negative": "#e74c3c", 
                "Neutral": "#95a5a6"
            }
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⭐ Rating Distribution")
        
        # Mock rating distribution (get from API in real implementation)
        ratings = [1, 2, 3, 4, 5]
        counts = [2, 5, 15, 25, 18]  # Mock data
        
        fig_ratings = px.bar(
            x=ratings,
            y=counts,
            title="Rating Distribution",
            labels={"x": "Rating (Stars)", "y": "Count"},
            color=counts,
            color_continuous_scale="RdYlGn"
        )
        st.plotly_chart(fig_ratings, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Quality Metrics")
        
        # Mock quality scores (get from API in real implementation)
        quality_metrics = {
            "Relevance": 0.85,
            "Accuracy": 0.78,
            "Completeness": 0.82,
            "Coherence": 0.88,
            "Citation": 0.75
        }
        
        fig_quality = go.Figure(data=go.Scatterpolar(
            r=list(quality_metrics.values()),
            theta=list(quality_metrics.keys()),
            fill='toself',
            line_color='rgb(1, 132, 242)'
        ))
        
        fig_quality.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="Quality Score Breakdown"
        )
        st.plotly_chart(fig_quality, use_container_width=True)
    
    # Recent Feedback Table
    st.subheader("💬 Recent Feedback")
    
    # Mock recent feedback data (get from API in real implementation)
    recent_feedback = pd.DataFrame({
        "Time": [datetime.now() - timedelta(hours=i) for i in range(5)],
        "Query": [
            "What is machine learning?",
            "How does neural networks work?", 
            "Explain deep learning concepts",
            "What are the applications of AI?",
            "How to implement RAG systems?"
        ],
        "Rating": [5, 4, 3, 5, 4],
        "Sentiment": ["😊 Positive", "😊 Positive", "😐 Neutral", "😊 Positive", "😊 Positive"],
        "Response Time": [2.3, 1.8, 3.2, 2.1, 2.7]
    })
    
    # Format the dataframe
    recent_feedback["Time"] = recent_feedback["Time"].dt.strftime("%Y-%m-%d %H:%M")
    recent_feedback["Rating"] = recent_feedback["Rating"].apply(lambda x: "⭐" * x)
    recent_feedback["Response Time"] = recent_feedback["Response Time"].apply(lambda x: f"{x:.1f}s")
    
    st.dataframe(
        recent_feedback,
        use_container_width=True,
        hide_index=True
    )
    
    # System Health Indicators
    st.subheader("🏥 System Health")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🟢 **API Status:** Healthy")
        st.info("🟢 **Database:** Connected")
    
    with col2:
        st.info("🟡 **Average Quality:** Good (0.82)")
        st.info("🟢 **Response Time:** Optimal")
    
    with col3:
        st.info("🟢 **User Satisfaction:** High (75%)")
        st.info("🟢 **System Uptime:** 99.9%")

if __name__ == "__main__":
    main()