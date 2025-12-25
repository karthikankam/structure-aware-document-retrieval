"""
Structure-Aware Embeddings Module
Combines text, visual, and structural features for document representation
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional
import numpy as np

# Directories
EMB_DIR = Path("./embeddings")
EMB_DIR.mkdir(parents=True, exist_ok=True)

class StructureAwareFusion(nn.Module):
    """
    Fuses text, visual, and structural embeddings into unified representation
    """
    def __init__(self, input_dim=512, output_dim=512):
        super().__init__()
        
        # Three separate encoders for each modality
        self.text_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.visual_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.struct_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Attention-based fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Final projection
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 3, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, text_emb, visual_emb=None, struct_emb=None):
        """
        Args:
            text_emb: (batch, 512) - text embedding
            visual_emb: (batch, 512) - visual embedding (optional)
            struct_emb: (batch, 512) - structural embedding (optional)
        
        Returns:
            fused: (batch, 512) - fused normalized embedding
        """
        # If only text is provided, use it three times (fallback)
        if visual_emb is None:
            visual_emb = text_emb.clone()
        if struct_emb is None:
            struct_emb = text_emb.clone()
        
        # Encode each modality
        text_enc = self.text_encoder(text_emb)
        visual_enc = self.visual_encoder(visual_emb)
        struct_enc = self.struct_encoder(struct_emb)
        
        # Stack for attention
        features = torch.stack([text_enc, visual_enc, struct_enc], dim=1)  # (batch, 3, 512)
        
        # Self-attention across modalities
        attn_out, _ = self.attention(features, features, features)  # (batch, 3, 512)
        
        # Flatten and fuse
        fused_features = attn_out.reshape(attn_out.shape[0], -1)  # (batch, 1536)
        fused = self.fusion(fused_features)
        
        # Normalize for similarity search
        return F.normalize(fused, dim=-1)


class StructuralFeatureEncoder(nn.Module):
    """
    Encodes structural features (layout, elements, etc.) into embeddings
    """
    def __init__(self, output_dim=512):
        super().__init__()
        
        # Encode categorical features
        self.element_type_emb = nn.Embedding(10, 64)  # 10 element types
        
        # Encode numeric features
        self.numeric_encoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        
        # Combine and project
        self.projection = nn.Sequential(
            nn.Linear(64 + 256, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, structure_features):
        """
        Args:
            structure_features: dict with keys:
                - num_pages: int
                - num_elements: int
                - element_types: dict {type: count}
                - layout_density: float
                - etc.
        
        Returns:
            struct_emb: (batch, 512) structural embedding
        """
        # This is a simplified version - customize based on your features
        batch_size = len(structure_features) if isinstance(structure_features, list) else 1
        
        # Create structural embedding (placeholder - customize based on your data)
        struct_emb = torch.randn(batch_size, 512)
        
        return struct_emb


def extract_structural_features(arxiv_id: str, structures_dir: Path = Path("./structures")) -> dict:
    """
    Extract structural features from processed document structure
    """
    struct_path = structures_dir / f"{arxiv_id}_structure.json"
    
    if not struct_path.exists():
        # Return default features if structure not available
        return {
            "num_pages": 0,
            "num_elements": 0,
            "element_types": {},
            "layout_density": 0.0,
            "has_equations": False,
            "has_figures": False,
            "has_tables": False
        }
    
    with open(struct_path, 'r') as f:
        structure = json.load(f)
    
    return structure.get('features', {})


def create_structure_embedding(features: dict, device: str = "cpu") -> torch.Tensor:
    """
    Create a simple structural embedding from features
    """
    # Simple encoding of structural features into 512-dim vector
    # This is a placeholder - replace with learned embeddings
    
    feature_vector = []
    
    # Numeric features
    feature_vector.append(features.get('num_pages', 0) / 100.0)  # Normalize
    feature_vector.append(features.get('num_elements', 0) / 1000.0)
    feature_vector.append(features.get('layout_density', 0.0))
    feature_vector.append(features.get('avg_elements_per_page', 0.0) / 100.0)
    
    # Binary features
    feature_vector.append(1.0 if features.get('has_equations', False) else 0.0)
    feature_vector.append(1.0 if features.get('has_figures', False) else 0.0)
    feature_vector.append(1.0 if features.get('has_tables', False) else 0.0)
    
    # Element type distribution
    element_types = features.get('element_types', {})
    for elem_type in ['title', 'section', 'paragraph', 'equation', 'figure', 'table', 'caption']:
        count = element_types.get(elem_type, 0)
        feature_vector.append(count / 100.0)  # Normalize
    
    # Pad to 512 dimensions
    while len(feature_vector) < 512:
        feature_vector.append(0.0)
    
    feature_vector = feature_vector[:512]  # Truncate if needed
    
    return torch.tensor(feature_vector, dtype=torch.float, device=device)


def save_embedding(arxiv_id: str, embedding: torch.Tensor, 
                   num_pages: Optional[int] = None, 
                   num_elements: Optional[int] = None):
    """
    Save embedding to JSON file
    """
    out = {
        "arxiv_id": arxiv_id,
        "embedding": embedding.squeeze(0).cpu().tolist()
    }
    
    if num_pages is not None:
        out["num_pages"] = num_pages
    if num_elements is not None:
        out["num_elements"] = num_elements
    
    output_path = EMB_DIR / f"{arxiv_id}_embedding.json"
    with open(output_path, "w") as f:
        json.dump(out, f)
    
    return output_path


def load_embedding(arxiv_id: str) -> Optional[torch.Tensor]:
    """
    Load embedding from JSON file
    """
    emb_path = EMB_DIR / f"{arxiv_id}_embedding.json"
    
    if not emb_path.exists():
        return None
    
    with open(emb_path, 'r') as f:
        data = json.load(f)
    
    # Handle both formats
    if "text_embedding" in data:
        emb = data["text_embedding"][0]
    elif "embedding" in data:
        emb = data["embedding"]
    else:
        return None
    
    return torch.tensor(emb, dtype=torch.float)


# Example usage
if __name__ == "__main__":
    # Initialize fusion model
    fusion_model = StructureAwareFusion()
    
    # Example: Create embeddings for a document
    arxiv_id = "2510.12362v1"
    
    # Load or create text embedding
    text_emb = torch.randn(1, 512)  # Placeholder
    
    # Extract structural features
    struct_features = extract_structural_features(arxiv_id)
    struct_emb = create_structure_embedding(struct_features).unsqueeze(0)
    
    # Create visual embedding (placeholder - use actual visual features)
    visual_emb = torch.randn(1, 512)
    
    # Fuse embeddings
    with torch.no_grad():
        fused_emb = fusion_model(text_emb, visual_emb, struct_emb)
    
    print(f"Text embedding shape: {text_emb.shape}")
    print(f"Visual embedding shape: {visual_emb.shape}")
    print(f"Structural embedding shape: {struct_emb.shape}")
    print(f"Fused embedding shape: {fused_emb.shape}")
    print(f"Fused embedding norm: {torch.norm(fused_emb).item():.4f}")
    
    # Save
    save_embedding(arxiv_id, fused_emb, num_pages=10, num_elements=150)
    print(f"Saved embedding for {arxiv_id}")