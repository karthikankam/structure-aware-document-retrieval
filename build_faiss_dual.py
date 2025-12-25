import json
import faiss
import numpy as np
import torch
from pathlib import Path
from dual_encoder_model import DualEncoder512

# Paths
EMB_DIR = Path("./embeddings")
FAISS_DIR = Path("./faiss")
MODELS_DIR = Path("./models")
MODEL_PATH = MODELS_DIR / "dual_encoder_512.pt"

# Create output directory
FAISS_DIR.mkdir(exist_ok=True)

print("Building FAISS index from embeddings...")

# Load the trained dual encoder
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

model = DualEncoder512().to(DEVICE)

if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"✅ Loaded trained model from {MODEL_PATH}")
else:
    print(f"⚠️  No trained model found at {MODEL_PATH}, using untrained model")

model.eval()

embeddings = []
metadata = []

# Process all embedding files
embedding_files = sorted(EMB_DIR.glob("*.json"))
print(f"Found {len(embedding_files)} embedding files")

if not embedding_files:
    print("❌ No embedding files found!")
    exit(1)

print("Encoding documents through dual encoder...")

with torch.no_grad():
    for f in embedding_files:
        data = json.load(open(f))
        
        # Handle both embedding formats
        if "text_embedding" in data:
            raw_emb = data["text_embedding"]
            # Handle nested list format
            if isinstance(raw_emb, list) and len(raw_emb) > 0 and isinstance(raw_emb[0], list):
                raw_emb = raw_emb[0]  # Extract first element if nested
        elif "embedding" in data:
            raw_emb = data["embedding"]
        else:
            print(f"Warning: No embedding found in {f.name}, skipping")
            continue
        
        # Convert to tensor and encode through document encoder
        doc_tensor = torch.tensor(raw_emb, dtype=torch.float).unsqueeze(0).to(DEVICE)
        encoded_emb = model.encode_doc(doc_tensor)
        
        # Convert back to numpy
        final_emb = encoded_emb.cpu().numpy()[0]
        
        embeddings.append(final_emb)
        metadata.append({
            "arxiv_id": data["arxiv_id"],
            "num_pages": data.get("num_pages", None),
            "num_elements": data.get("num_elements", None)
        })

if not embeddings:
    print("❌ No embeddings processed!")
    exit(1)

# Convert to numpy array
embeddings = np.array(embeddings).astype("float32")
print(f"Embeddings shape: {embeddings.shape}")

# Verify embeddings are normalized (should be ~1.0 for L2 normalized vectors)
norms = np.linalg.norm(embeddings, axis=1)
print(f"Embedding norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")

# Build FAISS index with Inner Product (cosine similarity for normalized vectors)
dimension = embeddings.shape[1]
print(f"Building FAISS index with dimension {dimension}...")

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save index and metadata
index_path = FAISS_DIR / "index.faiss"
meta_path = FAISS_DIR / "meta.json"

faiss.write_index(index, str(index_path))
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ FAISS index built and saved to {index_path}")
print(f"✅ Metadata saved to {meta_path}")
print(f"   Total documents indexed: {len(metadata)}")