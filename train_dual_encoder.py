import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json

from dual_encoder_model import DualEncoder512
from contrastive_loss import ContrastiveLoss
from dual_encoder_dataset import DualEncoderDataset

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDINGS_DIR = "./embeddings"
METADATA_DIR = "./metadata"
MODELS_DIR = "./models"
MODEL_PATH = Path(MODELS_DIR) / "dual_encoder_512.pt"

# Create models directory
Path(MODELS_DIR).mkdir(exist_ok=True)

print(f"Using device: {DEVICE}")

# Load dataset
dataset = DualEncoderDataset(
    embeddings_dir=EMBEDDINGS_DIR,
    metadata_dir=METADATA_DIR
)

loader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True,
    num_workers=0  # Set to 0 for debugging, increase for speed
)

# Load text encoder (for queries)
print("Loading text encoder...")
text_encoder = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2",
    device=DEVICE
)

# Freeze text encoder - we don't train it
for p in text_encoder.parameters():
    p.requires_grad = False

# Set to eval mode to avoid any training-specific behavior
text_encoder.eval()

# Initialize dual encoder model
print("Initializing dual encoder...")
model = DualEncoder512(
    query_dim=768,  # all-mpnet-base-v2 dimension
    doc_dim=512,    # your embedding dimension
    output_dim=512
).to(DEVICE)

# Loss and optimizer
criterion = ContrastiveLoss(temperature=0.05)
optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

# Training loop
print(f"\nStarting training for {5} epochs...")
print(f"Dataset size: {len(dataset)}")
print(f"Batches per epoch: {len(loader)}")

for epoch in range(5):
    model.train()
    total_loss = 0.0
    batch_count = 0

    for batch_idx, (queries, docs) in enumerate(loader):
        optimizer.zero_grad()

        # Encode queries using frozen text encoder
        # Use torch.inference_mode() instead of no_grad() for frozen encoder
        with torch.inference_mode():
            q_inference = text_encoder.encode(
                list(queries),
                convert_to_tensor=True,
                device=DEVICE,
                show_progress_bar=False
            )
        
        # Create a new tensor from the inference result
        # This breaks the inference mode connection
        q = q_inference.detach().clone()

        # Move document embeddings to device
        d = docs.to(DEVICE)

        # Encode through dual encoder (these WILL have gradients)
        q_emb = model.encode_query(q)
        d_emb = model.encode_doc(d)

        # Compute contrastive loss
        loss = criterion(q_emb, d_emb)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            print(f"  Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / batch_count
    print(f"Epoch {epoch+1}/5 | Average Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n✅ Dual encoder saved to {MODEL_PATH}")

# Save config for reference
config = {
    "model_type": "DualEncoder512",
    "query_dim": 768,
    "doc_dim": 512,
    "output_dim": 512,
    "text_encoder": "sentence-transformers/all-mpnet-base-v2",
    "training_samples": len(dataset),
    "final_loss": avg_loss
}

config_path = Path(MODELS_DIR) / "dual_encoder_config.json"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"✅ Config saved to {config_path}")