import streamlit as st
import os
import time
import json
import logging
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document as LCDocument
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from cerebras.cloud.sdk import Cerebras
from vector_store import VectorStore
from together import Together
from cachetools import TTLCache
import backoff
import sentry_sdk
from prometheus_client import start_http_server, Counter

# --- Configuration & Monitoring ---
load_dotenv()

# Initialize monitoring
sentry_sdk.init(os.getenv('SENTRY_DSN'))
start_http_server(8000)  # Prometheus metrics endpoint
REQUEST_COUNTER = Counter('app_requests', 'Total API requests')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize caching
response_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour cache

# --- UI CONFIG & STYLE (Retained from original) ---
st.set_page_config(page_title="DigiTwin RAG Forecast", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.cdnfonts.com/css/tw-cen-mt');
    * {
        font-family: 'Tw Cen MT', sans-serif !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"]::before {
        content: "▶";
        font-size: 1.3rem;
        margin-right: 0.4rem;
    }
    .logo-container {
        position: fixed;
        top: 5rem;
        right: 12rem;
        z-index: 9999;
    }
    </style>
""", unsafe_allow_html=True)

# Display logo
st.markdown(
    """
    <div class="logo-container">
        <img src="https://github.com/valonys/DigiTwin/blob/29dd50da95bec35a5abdca4bdda1967f0e5efff6/ValonyLabs_Logo.png?raw=true" width="70">
    </div>
    """,
    unsafe_allow_html=True
)

st.title("📊 DigiTwin - The Insp Nerdzx")

# --- Constants ---
USER_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/9904d9a0d445ab0488cf7395cb863cce7621d897/USER_AVATAR.png"
BOT_AVATAR = "https://raw.githubusercontent.com/achilela/vila_fofoka_analysis/991f4c6e4e1dc7a8e24876ca5aae5228bcdb4dba/Ataliba_Avatar.jpg"

# --- Enhanced System Prompts ---
PROMPTS = {
    "Daily Report Summarization": """You are DigiTwin, an expert inspector...""",
    # ... other prompts ...
}

# --- State Management ---
class AppState:
    @staticmethod
    def initialize():
        state_defaults = {
            "vectorstore": None,
            "chat_history": [],
            "model_intro_done": False,
            "current_model": None,
            "current_prompt": None,
            "last_processed": None
        }
        for key, val in state_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

AppState.initialize()

# --- Enhanced Document Processing ---
class DocumentProcessor:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def process_pdfs(_files):
        try:
            parsed_docs = []
            for f in _files:
                with st.spinner(f"Processing {f.name}..."):
                    reader = PdfReader(f)
                    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    parsed_docs.append(LCDocument(page_content=text, metadata={"name": f.name}))
            return parsed_docs
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            raise

    @staticmethod
    def build_vectorstore(_docs):
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cuda'} if torch.cuda.is_available() else {}
            )
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = []
            for i, doc in enumerate(_docs):
                for chunk in splitter.split_text(doc.page_content):
                    chunks.append(LCDocument(page_content=chunk, metadata={"source": f"doc_{i}"}))
            return FAISS.from_documents(chunks, embeddings)
        except Exception as e:
            logger.error(f"Vectorstore creation failed: {str(e)}")
            raise

# --- Enhanced Model Clients ---
class ModelClient:
    def __init__(self):
        self.together_client = Together(api_key=os.getenv('TOGETHER_API_KEY'))
        
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def generate_together(self, messages, model_name):
        try:
            response = self.together_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Together.ai API error: {str(e)}")
            raise

    # Add other model clients (OpenAI, Cerebras, etc.) with similar error handling

# --- Enhanced Response Generation ---
def generate_response(prompt):
    REQUEST_COUNTER.inc()
    cache_key = f"{prompt}_{st.session_state.current_model}_{st.session_state.current_prompt}"
    
    # Check cache first
    if cache_key in response_cache:
        logger.info("Serving response from cache")
        yield response_cache[cache_key]
        return
    
    messages = [{"role": "system", "content": PROMPTS[st.session_state.current_prompt]}]
    
    # Enhanced RAG Context
    if st.session_state.vectorstore:
        try:
            docs = st.session_state.vectorstore.similarity_search(prompt, k=5)
            context = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" 
                                 for doc in docs])
            messages.append({"role": "system", "content": f"Relevant Context:\n{context}"})
        except Exception as e:
            logger.warning(f"Vectorstore search failed: {str(e)}")
    
    messages.append({"role": "user", "content": prompt})
    full_response = ""
    model_client = ModelClient()

    try:
        if st.session_state.current_model == "XAI Inspector":
            # Enhanced Together.ai integration
            response = model_client.generate_together(
                messages,
                model_name="atalibamiguel21/amiguel-GM_Qwen1.8B_Finetune"
            )
            for word in response.split():
                full_response += word + " "
                yield f"<span style='font-family:Tw Cen MT'>{word} </span>"
                time.sleep(0.01)
        
        # Add other model integrations with proper error handling
        
        # Cache the successful response
        response_cache[cache_key] = full_response
        
    except Exception as e:
        error_msg = f"<span style='color:red'>⚠️ Error: {str(e)}</span>"
        logger.error(f"Generation failed: {str(e)}")
        yield error_msg
        raise

# --- Main UI Components ---
with st.sidebar:
    model_alias = st.selectbox("Choose your AI Agent", [
        "EE Smartest Agent", "JI Divine Agent", "EdJa-Valonys", "XAI Inspector", "Valonys Llama"
    ])
    
    file_type = st.radio("Select file type", ["PDF", "Excel"])
    
    if file_type == "PDF":
        uploaded_files = st.file_uploader("📄 Upload up to 10 PDF reports", type=["pdf"], accept_multiple_files=True)
    else:
        uploaded_files = st.file_uploader("📊 Upload Excel file", type=["xlsx", "xls"], accept_multiple_files=False)
    
    prompt_type = st.selectbox("Select the Task Type", list(PROMPTS.keys()))

# --- Document Processing ---
if uploaded_files:
    try:
        if file_type == "PDF":
            parsed_docs = DocumentProcessor.process_pdfs(uploaded_files)
            st.session_state.vectorstore = DocumentProcessor.build_vectorstore(parsed_docs)
            st.sidebar.success(f"{len(parsed_docs)} reports indexed.")
        else:
            # Enhanced Excel processing
            vector_store = VectorStore()
            excel_docs = vector_store.process_excel_to_documents(uploaded_files)
            if excel_docs:
                st.session_state.vectorstore = vector_store.process_documents(excel_docs)
                st.sidebar.success(f"{len(excel_docs)} notifications indexed.")
    except Exception as e:
        st.sidebar.error(f"Processing error: {str(e)}")
        logger.exception("Document processing failed")

# --- CHAT INTERFACE ---

# Agent Introduction Logic
if not st.session_state.model_intro_done or \
   st.session_state.current_model != model_alias or \
   st.session_state.current_prompt != prompt_type:
    
    agent_intros = {
        "EE Smartest Agent": "💡 EE Agent Activated — Pragmatic & Smart",
        "JI Divine Agent": "✨ JI Agent Activated — DeepSeek Reasoning",
        "EdJa-Valonys": "⚡ EdJa Agent Activated — Cerebras Speed",
        "XAI Inspector": "🔍 XAI Inspector — Qwen Custom Fine-tune",
        "Valonys Llama": "🦙 Valonys Llama — LLaMA3-Based Reasoning"
    }
    
    intro_message = agent_intros.get(model_alias, "🤖 AI Agent Activated")
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": intro_message,
        "timestamp": time.time()
    })
    st.session_state.model_intro_done = True
    st.session_state.current_model = model_alias
    st.session_state.current_prompt = prompt_type
    logger.info(f"Switched to model: {model_alias} with prompt: {prompt_type}")

# Display Chat History with Performance Metrics
for msg in st.session_state.chat_history:
    with st.chat_message(
        msg["role"], 
        avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
    ):
        # Add subtle timestamp for production debugging
        timestamp = ""
        if "timestamp" in msg:
            timestamp = f"<small style='color:#888;float:right;'>\
            {time.strftime('%H:%M:%S', time.localtime(msg['timestamp']))}</small>"
        
        st.markdown(f"{msg['content']}{timestamp}", unsafe_allow_html=True)

# Chat Input with Enhanced Features
if prompt := st.chat_input("Ask a summary or forecast about the reports..."):
    # Validate input before processing
    if len(prompt.strip()) < 3:
        st.warning("Please enter a more detailed question")
        st.stop()
    
    # Add user message to history with timestamp
    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt,
        "timestamp": time.time()
    })
    
    # Display user message
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    
    # Generate and stream response
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        response_placeholder = st.empty()
        full_response = ""
        start_time = time.time()
        
        try:
            for chunk in generate_response(prompt):
                full_response += chunk
                response_placeholder.markdown(
                    f"{full_response}▌", 
                    unsafe_allow_html=True
                )
            
            # Final render
            response_placeholder.markdown(full_response, unsafe_allow_html=True)
            
            # Log performance metrics
            duration = time.time() - start_time
            logger.info(f"Generated response in {duration:.2f}s for prompt: {prompt[:50]}...")
            
            # Add to history with metadata
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": time.time(),
                "metadata": {
                    "response_time": duration,
                    "model": st.session_state.current_model,
                    "prompt_type": st.session_state.current_prompt
                }
            })
            
        except Exception as e:
            error_msg = f"<span style='color:red'>⚠️ System Error: Please try again later</span>"
            response_placeholder.markdown(error_msg, unsafe_allow_html=True)
            logger.error(f"Response generation failed: {str(e)}")
            sentry_sdk.capture_exception(e)

if __name__ == "__main__":
    # Production configuration checks
    if not os.getenv('TOGETHER_API_KEY'):
        logger.warning("Together API key not set")
    if not os.getenv('HF_TOKEN'):
        logger.warning("HuggingFace token not set")
    
    logger.info("Application started")
