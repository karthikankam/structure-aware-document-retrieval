import json
import faiss
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer
from pathlib import Path
from dual_encoder_model import DualEncoder512

# Configuration
MODELS_DIR = Path("./models")
FAISS_DIR = Path("./faiss")
METADATA_DIR = Path("./metadata")

MODEL_PATH = MODELS_DIR / "dual_encoder_512.pt"
FAISS_INDEX = FAISS_DIR / "index.faiss"
FAISS_META = FAISS_DIR / "meta.json"

@st.cache_resource
def load_system():
    """Load all system components"""
    # Detect device inside the cached function
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    st.info(f"Loading system on {gpu_name}...")
    
    # Load text encoder for queries
    text_encoder = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2",
        device=device
    )
    
    # Load dual encoder model
    model = DualEncoder512().to(device)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        st.success(f"✅ Loaded model from {MODEL_PATH}")
    else:
        st.warning(f"⚠️ Model not found at {MODEL_PATH}, using untrained weights")
    model.eval()
    
    # Load FAISS index
    if not FAISS_INDEX.exists():
        st.error(f"❌ FAISS index not found at {FAISS_INDEX}")
        st.stop()
    index = faiss.read_index(str(FAISS_INDEX))
    
    # Load metadata
    if not FAISS_META.exists():
        st.error(f"❌ Metadata not found at {FAISS_META}")
        st.stop()
    meta = json.load(open(FAISS_META))
    
    st.success(f"✅ Loaded {index.ntotal} documents")
    
    return text_encoder, model, index, meta, device, gpu_name

# Load system
text_encoder, model, index, meta, DEVICE, GPU_NAME = load_system()

# UI
st.title("📄 Document Retrieval System")
st.markdown("Search through academic papers using semantic similarity")

# Sidebar
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
    show_scores = st.checkbox("Show similarity scores", value=True)
    show_metadata = st.checkbox("Show metadata", value=True)
    
    st.markdown("---")
    st.markdown(f"**Index Stats**")
    st.markdown(f"- Documents: {index.ntotal}")
    st.markdown(f"- Dimension: {index.d}")
    st.markdown(f"- Device: **{GPU_NAME}**")

# Search interface
query = st.text_input("Enter your search query:", placeholder="e.g., transformer models for document analysis")

if st.button("Search", type="primary") or (query and len(query) > 0):
    if not query:
        st.warning("Please enter a search query")
    else:
        with st.spinner("Searching..."):
            # Encode query
            with torch.no_grad():
                q = text_encoder.encode(
                    [query], 
                    convert_to_tensor=True, 
                    device=DEVICE,
                    show_progress_bar=False
                )
                q_emb = model.encode_query(q).cpu().numpy()
            
            # Search FAISS index
            scores, indices = index.search(q_emb, top_k)
            
            # Display results
            st.markdown("---")
            st.subheader(f"Top {top_k} Results")
            
            for rank, idx in enumerate(indices[0], start=1):
                with st.container():
                    col1, col2 = st.columns([0.1, 0.9])
                    
                    with col1:
                        st.markdown(f"### {rank}")
                    
                    with col2:
                        arxiv_id = meta[idx]['arxiv_id']
                        score = scores[0][rank-1]
                        
                        # Create clickable arXiv link
                        st.markdown(f"**[{arxiv_id}](https://arxiv.org/abs/{arxiv_id})**")
                        
                        if show_scores:
                            # Convert numpy float32 to Python float
                            score_float = float(score)
                            st.progress(min(score_float, 1.0))
                            st.caption(f"Similarity: {score_float:.4f}")
                        
                        if show_metadata:
                            # Load full metadata if available
                            meta_path = METADATA_DIR / f"{arxiv_id}.json"
                            if meta_path.exists():
                                full_meta = json.load(open(meta_path))
                                
                                if "title" in full_meta:
                                    st.markdown(f"*{full_meta['title']}*")
                                
                                if "abstract" in full_meta:
                                    with st.expander("Abstract"):
                                        st.write(full_meta["abstract"])
                                
                                if "authors" in full_meta:
                                    st.caption(f"Authors: {', '.join(full_meta['authors'][:3])}{'...' if len(full_meta['authors']) > 3 else ''}")
                            else:
                                # Show minimal metadata from FAISS meta
                                if meta[idx].get('num_pages'):
                                    st.caption(f"Pages: {meta[idx]['num_pages']}")
                                if meta[idx].get('num_elements'):
                                    st.caption(f"Elements: {meta[idx]['num_elements']}")
                    
                    st.markdown("---")

# Add some example queries
with st.expander("💡 Example Queries"):
    st.markdown("""
    - Document layout analysis with deep learning
    - OCR and text recognition using neural networks
    - Transformer models for document understanding
    - Table detection and extraction methods
    - Visual document analysis techniques
    """)
    
# Footer
st.markdown("---")
st.caption("Powered by FAISS, PyTorch, and Sentence Transformers")