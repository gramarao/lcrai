import streamlit as st
import requests
import pandas as pd
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from login import check_authentication, login_page

st.set_page_config(
    page_title="LCR Capital Partners - Admin",
    page_icon="🏢",
    layout="wide"
)

# Configuration
import os
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


from utils.styling import apply_branding, add_footer, add_header


# Then in your main code, after apply_branding():
apply_branding()
add_header()


# And at the end of your page:
add_footer()

def render_navigation_sidebar():
    """Navigation sidebar for dashboard"""
    with st.sidebar:
        username = st.session_state.get('username', 'User')
        user_role = st.session_state.get('user_role', 'user')
        
        st.markdown(f"👤 **{username}**")
        st.caption(f"Role: {user_role}")
        
        st.markdown("---")
        st.markdown("### Navigation")
        
        if st.button("💬 Chat", use_container_width=True):
            st.switch_page("01_🏠_Home.py")
        
        if st.button("📊 Dashboard", use_container_width=True):
            st.switch_page("pages/02_📊_Dashboard.py")
        
        if st.button("⚙️ Admin Panel", use_container_width=True, disabled=True):
            pass
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            from login import logout
            logout()


def load_styles_safely():
    """Load CSS safely for admin panel"""
    try:
        with open("styles.css", "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

def initialize_admin_session():
    """Initialize admin session state"""
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = {}
    if "feedback_data" not in st.session_state:
        st.session_state.feedback_data = []
    if "test_questions" not in st.session_state:
        st.session_state.test_questions = []

def main():
    # Check authentication
    if not check_authentication():
        login_page()
        return
    
    # Check if user has admin/superuser role
    # Restrict to admin/superuser only
    user_role = st.session_state.get("user_role", "user")
    if user_role not in ["admin", "superuser"]:
        st.error("🚫 Access Denied")
        st.stop()

    # Add sidebar navigation
    render_navigation_sidebar()

    load_styles_safely()
    initialize_admin_session()
    
    st.title("⚙️ RAG System Administration Panel")
    
    tab1, tab2, tab3 = st.tabs(["🔬 Model Comparison", "⚙️ System Settings", "📊 Analytics"])
    
    with tab1:
        model_comparison_interface()
    
    with tab2:
        system_settings_interface()
    
    with tab3:
        analytics_interface()

def model_comparison_interface():
    """Enhanced model comparison with side-by-side responses and feedback"""
    st.subheader("🔬 Model Performance Comparison")
    
    # Configuration section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Test Configuration")
        
        # Document selection for generating questions
        documents = get_available_documents()
        selected_doc = st.selectbox(
            "Select document for test questions:",
            options=["Manual Questions"] + documents,
            help="Choose a document to generate contextual questions"
        )
        
        # Question input method
        if selected_doc == "Manual Questions":
            questions_text = st.text_area(
                "Enter test questions (one per line):",
                value="What are the main topics discussed?\nWhat are the key findings?\nWhat recommendations are provided?",
                height=120
            )
            test_questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
        else:
            num_questions = st.slider("Number of questions to generate:", 3, 8, 5)
            if st.button("🎯 Generate Questions from Document"):
                with st.spinner("Generating contextual questions..."):
                    test_questions = generate_test_questions(selected_doc, num_questions)
                    if test_questions:
                        st.session_state.test_questions = test_questions
            
            test_questions = st.session_state.get('test_questions', [])
    
    with col2:
        st.markdown("### Models to Compare")
        available_models = get_available_models()
        selected_models = st.multiselect(
            "Select Gemini models:",
            options=available_models,
            default=available_models[:2] if len(available_models) >= 2 else available_models,
            help="Choose 2-4 models for optimal comparison"
        )
        
        if len(selected_models) > 4:
            st.warning("⚠️ More than 4 models may make the display cramped")
    
    # Display current questions
    if test_questions:
        st.markdown("### Test Questions")
        for i, question in enumerate(test_questions, 1):
            st.write(f"**Q{i}:** {question}")
    
    # Run comparison
    if test_questions and selected_models:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Run Model Comparison", type="primary"):
                run_side_by_side_comparison(test_questions, selected_models)
        
        with col2:
            if st.button("📊 Export Results") and st.session_state.comparison_results:
                export_results()
    
    # Display comparison results
    if st.session_state.comparison_results:
        display_side_by_side_results()

def run_side_by_side_comparison(questions: List[str], models: List[str]):
    """Run comparison and store results for side-by-side display"""
    results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_tests = len(questions) * len(models)
    current_test = 0
    
    for question_idx, question in enumerate(questions):
        results[question_idx] = {
            "question": question,
            "responses": {}
        }
        
        for model in models:
            current_test += 1
            progress = current_test / total_tests
            progress_bar.progress(progress)
            status_text.text(f"Testing {model}: {question[:50]}...")
            
            # Get response from model
            start_time = time.time()
            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"query": question, "model": model},
                    timeout=90
                )
                
                if response.status_code == 200:
                    data = response.json()
                    response_time = (time.time() - start_time) * 1000
                    
                    results[question_idx]["responses"][model] = {
                        "answer": data.get("answer", "No response"),
                        "quality": data.get("quality", {}),
                        "response_time": response_time,
                        "sources": data.get("sources", []),
                        "status": "success",
                        "raw_data": data
                    }
                else:
                    results[question_idx]["responses"][model] = {
                        "answer": f"Error: {response.status_code}",
                        "quality": {},
                        "response_time": 0,
                        "sources": [],
                        "status": "error",
                        "raw_data": {}
                    }
                    
            except Exception as e:
                results[question_idx]["responses"][model] = {
                    "answer": f"Exception: {str(e)}",
                    "quality": {},
                    "response_time": 0,
                    "sources": [],
                    "status": "error",
                    "raw_data": {}
                }
    
    progress_bar.empty()
    status_text.empty()
    
    st.session_state.comparison_results = results
    st.success(f"✅ Comparison completed! Tested {len(models)} models on {len(questions)} questions.")

def display_side_by_side_results():
    """Display comparison results in side-by-side format with feedback collection"""
    st.subheader("📊 Side-by-Side Comparison Results")
    
    # Summary metrics first
    show_summary_metrics()
    
    st.divider()
    
    for question_idx, result in st.session_state.comparison_results.items():
        question = result["question"]
        responses = result["responses"]
        
        st.markdown(f"### Question {question_idx + 1}")
        st.markdown(f"**{question}**")
        
        # Create columns for each model
        models = list(responses.keys())
        cols = st.columns(len(models))
        
        for col_idx, model in enumerate(models):
            with cols[col_idx]:
                response_data = responses[model]
                
                # Model header with status
                status_icon = "✅" if response_data["status"] == "success" else "❌"
                st.markdown(f"#### {status_icon} {model}")
                
                # Response content
                if response_data["status"] == "success":
                    st.markdown("**Response:**")
                    st.write(response_data["answer"])
                    
                    # Quality metrics
                    quality = response_data.get("quality", {})
                    if quality:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            score = quality.get("score", 0)
                            st.metric("Quality", f"{score:.3f}")
                        with col_b:
                            response_time = response_data.get("response_time", 0)
                            st.metric("Time (ms)", f"{response_time:.0f}")
                        
                        # Sources
                        sources = response_data.get("sources", [])
                        if sources:
                            st.caption(f"📚 {len(sources)} sources used")
                            with st.expander("View sources", expanded=False):
                                for source in sources[:3]:
                                    if isinstance(source, dict):
                                        name = source.get("source", "Unknown")
                                        sim = source.get("similarity", 0)
                                        st.write(f"• {name} ({sim:.3f})")
                else:
                    st.error(response_data["answer"])
                
                # Feedback collection
                st.markdown("---")
                st.markdown("**Feedback:**")
                
                feedback_key = f"rating_{question_idx}_{model}"
                rating = st.radio(
                    "Rate response:",
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: "⭐" * x,
                    key=feedback_key,
                    horizontal=True
                )
                
                comment_key = f"comment_{question_idx}_{model}"
                comment = st.text_area(
                    "Comments:",
                    key=comment_key,
                    height=60,
                    placeholder="What's good/bad about this response?"
                )
                
                # Store feedback
                feedback_data = {
                    "question_idx": question_idx,
                    "question": question,
                    "model": model,
                    "rating": rating,
                    "comment": comment,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Update feedback in session state
                existing_feedback = [f for f in st.session_state.feedback_data 
                                   if not (f["question_idx"] == question_idx and f["model"] == model)]
                existing_feedback.append(feedback_data)
                st.session_state.feedback_data = existing_feedback
        
        # Best response selection
        st.markdown("---")
        best_model = st.radio(
            f"**Best response for Question {question_idx + 1}:**",
            options=models,
            key=f"best_{question_idx}",
            horizontal=True
        )
        
        st.divider()
    
    # Model recommendation section
    st.subheader("🎯 Model Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.comparison_results:
            models_in_comparison = list(st.session_state.comparison_results[0]["responses"].keys())
            recommended_model = st.selectbox(
                "Select model for system:",
                options=models_in_comparison,
                help="This model will be used in the main chat interface"
            )
            
            if st.button("✅ Set as System Model", type="primary"):
                set_system_model(recommended_model)
    
    with col2:
        if st.button("💾 Save All Feedback"):
            save_feedback_data()
        
        if st.button("📊 Generate Report"):
            generate_comparison_report()

def show_summary_metrics():
    """Show summary statistics of the comparison"""
    if not st.session_state.comparison_results:
        return
    
    st.markdown("#### Summary Metrics")
    
    # Calculate summary statistics
    summary_data = []
    
    # Get all models from first question
    first_question = list(st.session_state.comparison_results.values())[0]
    models = list(first_question["responses"].keys())
    
    for model in models:
        total_score = 0
        total_time = 0
        success_count = 0
        
        for result in st.session_state.comparison_results.values():
            response = result["responses"].get(model, {})
            if response.get("status") == "success":
                quality = response.get("quality", {})
                total_score += quality.get("score", 0)
                total_time += response.get("response_time", 0)
                success_count += 1
        
        if success_count > 0:
            avg_score = total_score / success_count
            avg_time = total_time / success_count
        else:
            avg_score = avg_time = 0
        
        summary_data.append({
            "Model": model,
            "Avg Quality": f"{avg_score:.3f}",
            "Avg Time (ms)": f"{avg_time:.0f}",
            "Success Rate": f"{success_count}/{len(st.session_state.comparison_results)}"
        })
    
    # Display summary table
    df = pd.DataFrame(summary_data)
    st.dataframe(df, use_container_width=True)

def system_settings_interface():
    """System settings and model management - WITH FEEDBACK CONFIG"""
    st.subheader("⚙️ System Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Current System Model")
        
        current_model = get_current_system_model()
        available_models = get_available_models()
        
        new_model = st.selectbox(
            "Active model for chat interface:",
            options=available_models,
            index=available_models.index(current_model) if current_model in available_models else 0
        )
        
        if st.button("💾 Update System Model", type="primary"):
            if set_system_model(new_model):
                st.success(f"✅ System model updated to: {new_model}")
            else:
                st.error("❌ Failed to update system model")
        
        st.divider()
        
        # FEEDBACK CONFIGURATION
        st.markdown("### Feedback-Enhanced Retrieval")
        
        try:
            response = requests.get(f"{BACKEND_URL}/admin/feedback-config", timeout=5)
            if response.status_code == 200:
                config = response.json()
                
                feedback_enabled = st.checkbox(
                    "Enable feedback-enhanced retrieval",
                    value=config.get("feedback_enabled", True),
                    help="Use user feedback to improve search results"
                )
                
                feedback_weight = st.slider(
                    "Feedback weight:",
                    min_value=0.0,
                    max_value=1.0,
                    value=config.get("feedback_weight", 0.3),
                    step=0.05,
                    help="0.0 = ignore feedback, 1.0 = only feedback"
                )
                
                st.caption(f"""
                **Current setting:** {feedback_weight:.0%} feedback, {(1-feedback_weight):.0%} similarity
                
                - **0.0-0.2:** Minimal feedback influence (safe start)
                - **0.3-0.5:** Balanced approach (recommended)
                - **0.6-1.0:** Heavy feedback reliance (needs data)
                """)
                
                if st.button("💾 Update Feedback Settings"):
                    update_response = requests.post(
                        f"{BACKEND_URL}/admin/feedback-config",
                        json={
                            "feedback_enabled": feedback_enabled,
                            "feedback_weight": feedback_weight
                        },
                        timeout=5
                    )
                    
                    if update_response.status_code == 200:
                        st.success("✅ Feedback settings updated!")
                    else:
                        st.error("❌ Failed to update settings")
            else:
                st.warning("Could not load feedback configuration")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    with col2:
        st.markdown("### System Status")
        
        backend_status = check_backend_connectivity()
        status_icon = "🟢" if backend_status else "🔴"
        st.write(f"**Backend:** {status_icon}")
        
        models = get_available_models()
        st.write(f"**Models:** {len(models)} available")
        
        # Feedback stats
        try:
            response = requests.get(f"{BACKEND_URL}/feedback/analytics", timeout=5)
            if response.status_code == 200:
                analytics = response.json()
                st.write(f"**Feedback:** {analytics['summary']['total_feedback']} entries")
                st.write(f"**Positive Rate:** {analytics['summary']['positive_rate']:.1f}%")
        except:
            pass

def analytics_interface():
    """Analytics and feedback review - ENHANCED"""
    st.subheader("📊 System Analytics")
    
    # Fetch feedback analytics from backend
    try:
        response = requests.get(f"{BACKEND_URL}/feedback/analytics", timeout=10)
        
        if response.status_code == 200:
            analytics = response.json()
            
            # Summary metrics
            st.markdown("### Feedback Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Feedback", 
                    analytics["summary"]["total_feedback"]
                )
            
            with col2:
                st.metric(
                    "Positive Rate", 
                    f"{analytics['summary']['positive_rate']:.1f}%"
                )
            
            with col3:
                st.metric(
                    "Avg Rating", 
                    f"{analytics['summary']['avg_rating']:.2f}/5"
                )
            
            with col4:
                st.metric(
                    "Avg Relevance", 
                    f"{analytics['summary']['avg_relevance']:.3f}"
                )
            
            # Rating distribution
            if analytics["rating_distribution"]:
                st.markdown("### Rating Distribution")
                import pandas as pd
                df = pd.DataFrame(analytics["rating_distribution"])
                st.bar_chart(df.set_index("rating"))
            
            # Recent negative feedback
            if analytics["recent_negative_feedback"]:
                st.markdown("### Recent Issues (Low Ratings)")
                for fb in analytics["recent_negative_feedback"]:
                    with st.expander(f"⭐ {fb['rating']}/5 - {fb['question']}", expanded=False):
                        st.write(f"**Comment:** {fb['comment']}")
                        st.caption(f"Query ID: {fb['query_id']}")
                        st.caption(f"Time: {fb['timestamp']}")
        else:
            st.error("Could not fetch analytics")
            
    except Exception as e:
        st.error(f"Error loading analytics: {str(e)}")
    
    # Model comparison feedback (existing code)
    if st.session_state.feedback_data:
        st.markdown("### Model Comparison Feedback")
        df = pd.DataFrame(st.session_state.feedback_data)
        
        if not df.empty:
            avg_ratings = df.groupby("model")["rating"].agg(["mean", "count"]).reset_index()
            avg_ratings.columns = ["Model", "Average Rating", "Total Ratings"]
            avg_ratings["Average Rating"] = avg_ratings["Average Rating"].round(2)
            
            st.dataframe(avg_ratings, use_container_width=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_available_documents() -> List[str]:
    """Get available documents from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data if isinstance(data, list) else data.get("documents", [])
    except:
        pass
    return []

def get_available_models() -> List[str]:
    """Get available Gemini models"""
    return [
        "gemini-2.5-flash",        # Current default - good balance
        "gemini-2.5-pro",          # Best quality
        "gemini-1.5-flash-latest", # Budget option (10x cheaper)
        "gemini-1.5-pro-latest"    # Previous gen
    ]

def generate_test_questions(document: str, num_questions: int) -> List[str]:
    """Generate test questions from document"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/admin/generate-questions",
            json={"document": document, "num_questions": num_questions},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("questions", [])
        else:
            st.error(f"Failed to generate questions: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return []

def check_backend_connectivity() -> bool:
    """Check if backend is accessible"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_current_system_model() -> str:
    """Get currently selected model from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/admin/selected-model", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("model", "gemini-2.5-flash")
    except:
        pass
    return "gemini-2.5-flash"

def set_system_model(model: str) -> bool:
    """Set system model via backend"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/admin/set-model",
            json={"model": model},
            timeout=5
        )
        return response.status_code == 200
    except:
        return False

def save_feedback_data():
    """Save feedback data to backend"""
    if not st.session_state.feedback_data:
        st.warning("No feedback data to save")
        return
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/admin/feedback",
            json={"feedback": st.session_state.feedback_data},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Saved {result.get('saved_entries', 0)} feedback entries!")
            st.session_state.feedback_data = []
        else:
            st.error(f"Failed to save feedback: {response.status_code}")
    except Exception as e:
        st.error(f"Error saving feedback: {str(e)}")

def export_results():
    """Export comparison results to JSON"""
    if not st.session_state.comparison_results:
        st.warning("No results to export")
        return
    
    try:
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "comparison_results": st.session_state.comparison_results,
            "feedback_data": st.session_state.feedback_data
        }
        
        json_str = json.dumps(export_data, indent=2)
        
        st.download_button(
            label="📥 Download Results (JSON)",
            data=json_str,
            file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        st.success("✅ Results ready for download!")
    except Exception as e:
        st.error(f"Error exporting results: {str(e)}")

def generate_comparison_report():
    """Generate a markdown report of comparison results"""
    if not st.session_state.comparison_results:
        st.warning("No results to report")
        return
    
    try:
        report_lines = [
            "# Model Comparison Report",
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n**Questions Tested:** {len(st.session_state.comparison_results)}",
            "\n---\n"
        ]
        
        first_question = list(st.session_state.comparison_results.values())[0]
        models = list(first_question["responses"].keys())
        
        report_lines.append("## Summary Statistics\n")
        
        for model in models:
            total_score = 0
            total_time = 0
            success_count = 0
            
            for result in st.session_state.comparison_results.values():
                response = result["responses"].get(model, {})
                if response.get("status") == "success":
                    quality = response.get("quality", {})
                    total_score += quality.get("score", 0)
                    total_time += response.get("response_time", 0)
                    success_count += 1
            
            if success_count > 0:
                avg_score = total_score / success_count
                avg_time = total_time / success_count
                
                report_lines.append(f"\n### {model}")
                report_lines.append(f"- Average Quality: {avg_score:.3f}")
                report_lines.append(f"- Average Time: {avg_time:.0f}ms")
                report_lines.append(f"- Success Rate: {success_count}/{len(st.session_state.comparison_results)}")
        
        report_lines.append("\n---\n## Detailed Results\n")
        
        for question_idx, result in st.session_state.comparison_results.items():
            report_lines.append(f"\n### Question {question_idx + 1}")
            report_lines.append(f"\n**{result['question']}**\n")
            
            for model, response_data in result["responses"].items():
                report_lines.append(f"\n#### {model}")
                
                if response_data["status"] == "success":
                    report_lines.append(f"\n{response_data['answer']}\n")
                    quality = response_data.get("quality", {})
                    report_lines.append(f"- Quality Score: {quality.get('score', 0):.3f}")
                    report_lines.append(f"- Response Time: {response_data.get('response_time', 0):.0f}ms")
                else:
                    report_lines.append(f"\n❌ {response_data['answer']}\n")
        
        report_text = "\n".join(report_lines)
        
        st.download_button(
            label="📄 Download Report (Markdown)",
            data=report_text,
            file_name=f"model_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
        
        st.success("✅ Report ready for download!")
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")


if __name__ == "__main__":
    main()
