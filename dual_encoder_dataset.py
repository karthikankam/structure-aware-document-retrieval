import json
import torch
from torch.utils.data import Dataset
from pathlib import Path

class DualEncoderDataset(Dataset):
    def __init__(self, embeddings_dir="./embeddings", metadata_dir="./metadata"):
        self.embeddings_dir = Path(embeddings_dir)
        self.metadata_dir = Path(metadata_dir)
        
        # Get all embedding files
        self.embedding_files = list(self.embeddings_dir.glob("*.json"))
        
        # Extract arxiv IDs from filenames
        self.ids = [f.stem for f in self.embedding_files]
        
        print(f"Loaded {len(self.ids)} papers for training")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        arxiv_id = self.ids[idx]
        
        # Load metadata
        meta_path = self.metadata_dir / f"{arxiv_id}.json"
        if meta_path.exists():
            meta = json.load(open(meta_path))
            query = meta.get("title", "") + ". " + meta.get("abstract", "")
        else:
            # Fallback if metadata doesn't exist
            query = f"Document {arxiv_id}"
        
        # Load embedding - handle both formats
        emb_path = self.embeddings_dir / f"{arxiv_id}.json"
        emb_data = json.load(open(emb_path))
        
        # Support both "embedding" and "text_embedding" keys
        if "text_embedding" in emb_data:
            doc_emb = torch.tensor(emb_data["text_embedding"][0], dtype=torch.float)
        elif "embedding" in emb_data:
            doc_emb = torch.tensor(emb_data["embedding"], dtype=torch.float)
        else:
            raise ValueError(f"No embedding found in {emb_path}")
        
        # Ensure it's the right dimension (512)
        if doc_emb.shape[0] != 512:
            print(f"Warning: {arxiv_id} has embedding dim {doc_emb.shape[0]}, expected 512")
        
        return query, doc_emb