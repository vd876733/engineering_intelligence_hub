import sys
import os
import streamlit as st
import time

# --- API Key Bootstrap ---
# load_dotenv() MUST run before any os.getenv() or module import that reads env vars.
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY", "").strip("'\" ")
if not api_key:
    st.error(
        "GEMINI_API_KEY is not configured in environment variables or .env file. "
        "Add it to your .env file:\n\n  GEMINI_API_KEY=<your-key>"
    )
    st.stop()   # Halt rendering gracefully; no Python traceback shown to the user.

# Ensure src modules are in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.parser.ast_chunker import parse_repository
from src.agent.code_agent import VECTOR_STORE, get_agent_response
from google.genai.errors import APIError, ClientError
import asyncio

st.set_page_config(page_title="Engineering Intelligence Hub", page_icon="🧠", layout="wide")

# Inject Custom CSS for the Deep Charcoal and Slate Gray theme
st.markdown(
    """
    <style>
    /* Main Backgrounds */
    .stApp {
        background-color: #121212 !important;
        color: #E0E0E0 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #1A1A1D !important;
        border-right: 1px solid #2D2D30;
    }
    
    /* System Control Sidebar styling */
    .sidebar-header {
        color: #87CEFA; /* Light sky blue */
        font-weight: 800;
        font-size: 1.5rem;
        margin-bottom: 20px;
    }
    .system-control-title {
        color: #A0A0A5;
        font-size: 0.85rem;
        letter-spacing: 2px;
        margin-bottom: 20px;
        text-transform: uppercase;
    }
    
    /* Pulsing Status */
    .status-pulse {
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: #00FF00;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 8px #00FF00;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 0, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(0, 255, 0, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 0, 0); }
    }
    
    /* Data Badge */
    .data-badge {
        background-color: #2D2D30;
        padding: 6px 12px;
        border-radius: 6px;
        border: 1px solid #3E3E42;
        font-family: monospace;
        color: #4DA8DA;
        font-weight: bold;
        display: inline-block;
        margin-top: 5px;
    }
    
    /* Extra Metrics */
    .metric-label {
        color: #A0A0A5;
        font-size: 0.8rem;
        text-transform: uppercase;
        margin-top: 15px;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #E0E0E0;
        font-size: 0.9rem;
        font-family: monospace;
    }

    /* Main Area Headers */
    .main-title {
        font-family: sans-serif;
        font-size: 2.5rem;
        font-weight: 900;
        color: #E0E0E0;
        margin-bottom: 30px;
        letter-spacing: -0.5px;
        border-bottom: 1px solid #2D2D30;
        padding-bottom: 15px;
    }
    
    /* User Chat Pill */
    .user-pill {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 20px;
        padding: 15px 25px;
        margin-bottom: 30px;
        display: flex;
        align-items: center;
        box-shadow: 0 0 10px rgba(255, 165, 0, 0.1);
    }
    .user-avatar {
        background-color: #FF8C00;
        color: #FFF;
        border-radius: 50%;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 15px;
        flex-shrink: 0;
    }
    .user-text {
        color: #E0E0E0;
        font-size: 1.1rem;
    }
    
    /* Response Container */
    .response-container {
        background-color: #161618;
        border: 1px solid #005A9E; /* Engineering Blue */
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 15px rgba(0, 90, 158, 0.2);
    }
    
    /* Code Metadata Headers */
    .metadata-header {
        color: #FF8C00; /* Orange */
        font-family: monospace;
        font-size: 0.85rem;
        font-weight: bold;
        margin-bottom: 10px;
        border-bottom: 1px dashed #333;
        padding-bottom: 5px;
    }
    .metadata-item {
        margin-right: 15px;
    }
    
    /* Terminal Emulator */
    .terminal-widget {
        background-color: #0D0D0D !important;
        border: 1px solid #333;
        border-radius: 6px;
        padding: 15px;
        font-family: "Consolas", "Courier New", monospace !important;
    }
    /* Syntax Highlight Overrides for Terminal */
    .terminal-widget pre {
        background-color: transparent !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Agent Explanation Text */
    .agent-explanation {
        color: #C0C0C0;
        font-size: 1rem;
        line-height: 1.6;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource(show_spinner=False)
def initialize_repository():
    """Parse and index the repository into the in-memory Qdrant store."""
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    src_path = os.path.join(repo_path, "src")
    chunks = parse_repository(src_path)
    VECTOR_STORE.index(chunks)
    return True

# Initialize repository on startup
with st.spinner("Initializing system and loading models..."):
    initialize_repository()

# --- Sidebar (System Control) ---
with st.sidebar:
    st.markdown('<div class="sidebar-header">🧠 Eng Intel Hub</div>', unsafe_allow_html=True)
    st.markdown('<div class="system-control-title">System Control</div>', unsafe_allow_html=True)
    
    st.markdown('<div><span class="status-pulse"></span><span style="color:#E0E0E0; font-weight:bold;">Repository Status: Active</span></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="metric-label">Total Indexed AST Nodes</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="data-badge">{VECTOR_STORE.count()}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    top_k = st.slider("Top-K Code Blocks to Retrieve", min_value=1, max_value=10, value=3)
    
    st.markdown("---")
    st.markdown('<div class="metric-label">Embedding Model Status</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-value" style="color: #4DA8DA;">🟢 BAAI/bge-small-en-v1.5 (Loaded)</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="metric-label">Vector Store</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-value">Qdrant (:memory:)</div>', unsafe_allow_html=True)

# --- Main Chat Area ---
st.markdown('<div class="main-title">CODEBASE ASSISTANT: RAG Insights</div>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f"""
            <div class="user-pill">
                <div class="user-avatar">U</div>
                <div class="user-text">{message["content"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif message["role"] == "assistant":
        st.markdown('<div class="response-container">', unsafe_allow_html=True)
        if "context" in message and message["context"]:
            with st.expander("View Retrieved Context", expanded=False):
                for ctx in message["context"]:
                    st.markdown(
                        f"""
                        <div class="metadata-header">
                            <span class="metadata-item">LOCATED: {ctx['file_path']}</span> | 
                            <span class="metadata-item">FUNCTION: {ctx['function_name']}</span> | 
                            <span class="metadata-item">LINES: {ctx['start_line']}-{ctx['end_line']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown('<div class="terminal-widget">', unsafe_allow_html=True)
                    st.code(ctx['snippet'], language='python')
                    st.markdown('</div><br>', unsafe_allow_html=True)
            
        st.markdown(f'<div class="agent-explanation">{message["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# React to user input
if prompt := st.chat_input("Ask a question about the codebase..."):
    st.markdown(
        f"""
        <div class="user-pill">
            <div class="user-avatar">U</div>
            <div class="user-text">{prompt}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    st.markdown('<div class="response-container">', unsafe_allow_html=True)
    
    # 1. Search for context
    start_time = time.time()
    results = VECTOR_STORE.search(prompt, top_k=top_k)
    latency = time.time() - start_time
    
    context_data = []
    
    # 2. Render retrieved AST source blocks in expander cards
    if results:
        with st.expander("View Retrieved Context", expanded=False):
            for r in results:
                path = r.get("file_path", "unknown")
                start = r.get("start_line", "?")
                end = r.get("end_line", "?")
                func = r.get("function_name", "unknown")
                snippet = r.get("code_snippet", "")
                
                context_data.append({
                    "file_path": path,
                    "start_line": start,
                    "end_line": end,
                    "function_name": func,
                    "snippet": snippet
                })
                
                st.markdown(
                    f"""
                    <div class="metadata-header">
                        <span class="metadata-item">LOCATED: {path}</span> | 
                        <span class="metadata-item">FUNCTION: {func}</span> | 
                        <span class="metadata-item">LINES: {start}-{end}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown('<div class="terminal-widget">', unsafe_allow_html=True)
                st.code(snippet, language='python')
                st.markdown('</div><br>', unsafe_allow_html=True)
    
    # 3. Agent response with spinner and exception handling
    full_response = ""
    with st.spinner("Analyzing codebase AST and generating response..."):
        try:
            full_response = asyncio.run(get_agent_response(prompt, context_data))
            st.markdown(f'<div class="agent-explanation">{full_response}</div>', unsafe_allow_html=True)
        except ClientError as e:
            if e.code == 429:
                full_response = (
                    "⚠️ **Rate limit reached (429).** "
                    "The free-tier allows only a few requests per minute. "
                    "Please wait **15–60 seconds** and try again."
                )
                st.warning(full_response, icon="⏳")
            else:
                full_response = f"API error ({e.code}): {str(e)}"
                st.error(full_response)
        except Exception as e:
            full_response = f"I encountered an error while communicating with the agent: {str(e)}"
            st.error(full_response)
            
    st.markdown('</div>', unsafe_allow_html=True)  # close response-container
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "context": context_data
    })
