#!/usr/bin/env python3
"""
Streamlit Chat Interface for Memvid
A user-friendly web interface for interacting with QR code video memories
"""

import streamlit as st
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime
import json

# Add the parent directory to the path so we can import memvid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memvid.chat import MemvidChat
from memvid.encoder import MemvidEncoder
from memvid.config import get_default_config, get_codec_parameters

def get_video_file_type(codec=None):
    """Get the video file extension for the given codec"""
    if codec is None:
        config = get_default_config()
        codec = config["codec"]
    
    codec_params = get_codec_parameters(codec)
    return codec_params["video_file_type"]

# Page configuration
st.set_page_config(
    page_title="Memvid Chat",
    page_icon="💾",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_instance' not in st.session_state:
        st.session_state.chat_instance = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'memory_loaded' not in st.session_state:
        st.session_state.memory_loaded = False
    if 'video_file' not in st.session_state:
        st.session_state.video_file = None
    if 'index_file' not in st.session_state:
        st.session_state.index_file = None
    if 'memory_just_created' not in st.session_state:
        st.session_state.memory_just_created = False

def sidebar_config():
    """Sidebar configuration for memory management and settings"""
    with st.sidebar:
        st.title("🧠 Memory Configuration")
        
        # Memory file selection
        st.subheader("📂 Select Memory Files")
        st.write("Browse and select your existing memory files or create a new memory below.")
        
        # Show special message if memory was just created
        if st.session_state.memory_just_created:
            st.success("🎉 Memory created! Files are now selected.")
            st.session_state.memory_just_created = False  # Reset flag
        
        # File browsers for memory files
        video_file = st.file_uploader(
            "Select Video Memory File",
            type=[get_video_file_type()],
            help="Choose your QR code video memory file"
        )
        
        index_file = st.file_uploader(
            "Select Index File",
            type=["json"],
            help="Choose your memory index JSON file"
        )
        
        # Use uploaded files or fall back to session state paths
        if video_file:
            video_path = video_file.name
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{get_video_file_type()}") as tmp_video:
                tmp_video.write(video_file.getvalue())
                video_path = tmp_video.name
        else:
            video_path = st.session_state.video_file
            
        if index_file:
            index_path = index_file.name
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_index:
                tmp_index.write(index_file.getvalue())
                index_path = tmp_index.name
        else:
            index_path = st.session_state.index_file
        
        # Check if files exist
        video_exists = os.path.exists(video_path) if video_path else False
        index_exists = os.path.exists(index_path) if index_path else False
        
        # Display file status and load button
        both_files_available = (video_file and index_file) or (video_exists and index_exists)
        
        if both_files_available:
            st.success("✅ Memory files ready!")
            
            # Check if API key is provided for the button state
            current_api_key = getattr(st.session_state, 'api_key', None)
            has_api_key = current_api_key and current_api_key.strip()
            
            if not has_api_key:
                st.warning("⚠️ Please enter an API key in LLM Settings to load memory")
            
            if st.button("🧠 Load Memory", 
                        disabled=not has_api_key, 
                        help="Load the memory files and initialize chat" if has_api_key else "Enter an API key first",
                        use_container_width=True):
                with st.spinner("Loading memory..."):
                    load_memory(video_path, index_path)
        else:
            if not video_file and not st.session_state.video_file:
                st.info("📁 Please select a video memory file")
            if not index_file and not st.session_state.index_file:
                st.info("📄 Please select an index JSON file")
        
        st.divider()
        
        # Create new memory
        st.subheader("📝 Create New Memory")
        
        # File upload for creating memory
        uploaded_file = st.file_uploader(
            "Upload document to create memory",
            type=['txt', 'pdf', 'md'],
            help="Upload a text file, PDF, or Markdown file to create a new QR video memory"
        )
        
        if uploaded_file is not None:
            if st.button("Create Memory from Upload"):
                create_memory_from_upload(uploaded_file)
        
        st.divider()
        
        # LLM Configuration
        st.subheader("🤖 LLM Settings")
        
        llm_provider = st.selectbox(
            "LLM Provider",
            ["openai", "google", "anthropic"],
            index=0,
            help="Choose your preferred language model provider"
        )
        
        # Get current API key from session state or empty string
        current_api_key = getattr(st.session_state, 'api_key', '')
        
        api_key = st.text_input(
            "API Key",
            value=current_api_key,
            type="password",
            help="Enter your API key for the selected provider"
        )
        
        # Store in session state
        st.session_state.llm_provider = llm_provider
        st.session_state.api_key = api_key
        
        st.divider()
        
        # Memory info
        if st.session_state.memory_loaded and st.session_state.chat_instance:
            st.subheader("📊 Memory Stats")
            stats = st.session_state.chat_instance.get_stats()
            st.json(stats)

def load_memory(video_path: str, index_path: str):
    """Load memory files and initialize chat"""
    try:
        # Get LLM settings from session state
        llm_provider = getattr(st.session_state, 'llm_provider', 'openai')
        api_key = getattr(st.session_state, 'api_key', None)
        
        st.info(f"Loading memory with {llm_provider}...")
        st.info(f"Video: {video_path}")
        st.info(f"Index: {index_path}")
        
        # Initialize chat instance
        chat_instance = MemvidChat(
            video_file=video_path,
            index_file=index_path,
            llm_provider=llm_provider,
            llm_api_key=api_key
        )
        
        # Start session
        chat_instance.start_session()
        
        # Store in session state
        st.session_state.chat_instance = chat_instance
        st.session_state.video_file = video_path
        st.session_state.index_file = index_path
        st.session_state.memory_loaded = True
        st.session_state.messages = []  # Clear previous messages
        
        st.success(f"✅ Memory loaded successfully! Using {llm_provider} LLM.")
        st.rerun()
        
    except Exception as e:
        import traceback
        st.error(f"❌ Error loading memory: {str(e)}")
        st.error(f"Full error: {traceback.format_exc()}")
        st.error("Please check your API key and file paths.")

def create_memory_from_upload(uploaded_file):
    """Create a new memory from uploaded file"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Generate output filenames with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"memory_{timestamp}.{get_video_file_type()}"
        index_filename = f"memory_index_{timestamp}.json"
        
        video_path = output_dir / video_filename
        index_path = output_dir / index_filename
        
        # Initialize encoder from file
        config = get_default_config()
        
        with st.spinner("Creating QR code video memory... This may take a few minutes."):
            # Create encoder from file and build video
            encoder = MemvidEncoder.from_file(tmp_path, config=config)
            build_stats = encoder.build_video(str(video_path), str(index_path), show_progress=True)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        st.success(f"✅ Memory created successfully!")
        st.info(f"Video: {video_path}")
        st.info(f"Index: {index_path}")
        
        # Update session state with new file paths
        st.session_state.video_file = str(video_path)
        st.session_state.index_file = str(index_path)
        st.session_state.memory_just_created = True
        
        # Auto-load the new memory
        current_api_key = getattr(st.session_state, 'api_key', None)
        has_api_key = current_api_key and current_api_key.strip()
        
        if not has_api_key:
            st.warning("⚠️ Please enter an API key in LLM Settings to load memory")
        
        if st.button("🧠 Load New Memory", 
                    disabled=not has_api_key,
                    help="Load the newly created memory and start chatting" if has_api_key else "Enter an API key first",
                    use_container_width=True):
            load_memory(str(video_path), str(index_path))
        
        # Trigger a rerun to update the text inputs with new paths
        st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error creating memory: {str(e)}")
        # Clean up temporary file if it exists
        try:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except:
            pass

def main_chat_interface():
    """Main chat interface"""
    st.title("💬 Memvid Chat Interface")
    
    if not st.session_state.memory_loaded:
        st.info("👈 Please load a memory from the sidebar to start chatting!")
        st.markdown("""
        ### Getting Started:
        1. **Load existing memory**: Enter paths to your video and index files in the sidebar
        2. **Create new memory**: Upload a document to create a new QR video memory
        3. **Configure LLM**: Choose your provider and enter API key
        4. **Start chatting**: Ask questions about your memory content!
        """)
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your memory..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get response from MemvidChat
                    response = st.session_state.chat_instance.chat(prompt)
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

def export_conversation():
    """Export conversation to JSON"""
    if st.session_state.messages:
        conversation_data = {
            'timestamp': datetime.now().isoformat(),
            'video_file': st.session_state.video_file,
            'index_file': st.session_state.index_file,
            'llm_provider': getattr(st.session_state, 'llm_provider', 'unknown'),
            'messages': st.session_state.messages
        }
        
        # Create download button
        json_str = json.dumps(conversation_data, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Export Conversation",
            data=json_str,
            file_name=f"memvid_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def main():
    """Main application function"""
    initialize_session_state()
    
    # Sidebar
    sidebar_config()
    
    # Main interface
    main_chat_interface()
    
    # Footer with export option
    if st.session_state.messages:
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                if st.session_state.chat_instance:
                    st.session_state.chat_instance.clear_history()
                st.rerun()
        
        with col2:
            export_conversation()
        
        with col3:
            st.write(f"💬 {len(st.session_state.messages)} messages")

if __name__ == "__main__":
    main()