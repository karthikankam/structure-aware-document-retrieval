import torch
import torch.nn as nn
import torch.nn.functional as F

class DualEncoder512(nn.Module):
    """
    Dual encoder that maps queries and documents to 512-dim space
    Query: 768-dim (from all-mpnet-base-v2) -> 512-dim
    Document: 512-dim (pre-computed) -> 512-dim
    """
    def __init__(self, query_dim=768, doc_dim=512, output_dim=512):
        super().__init__()
        
        # Query encoder: maps text embeddings to shared space
        self.query_proj = nn.Sequential(
            nn.Linear(query_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
        # Document encoder: refines pre-computed embeddings
        self.doc_proj = nn.Sequential(
            nn.Linear(doc_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )

    def encode_query(self, x):
        """Encode query text embeddings"""
        x = self.query_proj(x)
        return F.normalize(x, dim=-1)

    def encode_doc(self, x):
        """Encode document embeddings"""
        x = self.doc_proj(x)
        return F.normalize(x, dim=-1)
    
    def forward(self, query_emb, doc_emb):
        """Forward pass for training"""
        q = self.encode_query(query_emb)
        d = self.encode_doc(doc_emb)
        return q, d