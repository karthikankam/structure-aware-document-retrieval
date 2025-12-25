"""
Complete workflow script for document retrieval system
Runs all steps: training, index building, and testing
"""

import subprocess
import sys
from pathlib import Path
import json

def run_command(cmd, description):
    """Run a command and report status"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        return False

def check_prerequisites():
    """Check if required directories and files exist"""
    print("\n" + "="*60)
    print("CHECKING PREREQUISITES")
    print("="*60)
    
    checks = []
    
    # Check embeddings
    emb_dir = Path("./embeddings")
    if emb_dir.exists():
        emb_count = len(list(emb_dir.glob("*.json")))
        print(f"✅ Embeddings directory: {emb_count} files")
        checks.append(emb_count > 0)
    else:
        print(f"❌ Embeddings directory not found")
        checks.append(False)
    
    # Check metadata
    meta_dir = Path("./metadata")
    if meta_dir.exists():
        meta_count = len(list(meta_dir.glob("*.json")))
        print(f"✅ Metadata directory: {meta_count} files")
        checks.append(meta_count > 0)
    else:
        print(f"❌ Metadata directory not found")
        checks.append(False)
    
    # Check Python files
    required_files = [
        "dual_encoder_model.py",
        "dual_encoder_dataset.py",
        "contrastive_loss.py",
        "train_dual_encoder.py",
        "build_faiss_dual.py"
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
            checks.append(True)
        else:
            print(f"❌ {file} not found")
            checks.append(False)
    
    return all(checks)

def check_trained_model():
    """Check if model is already trained"""
    model_path = Path("./models/dual_encoder_512.pt")
    return model_path.exists()

def check_faiss_index():
    """Check if FAISS index is built"""
    index_path = Path("./faiss/index.faiss")
    meta_path = Path("./faiss/meta.json")
    return index_path.exists() and meta_path.exists()

def main():
    print("="*60)
    print("DOCUMENT RETRIEVAL SYSTEM - COMPLETE WORKFLOW")
    print("="*60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return 1
    
    # Step 1: Train dual encoder (if not already trained)
    if check_trained_model():
        print("\n✅ Model already trained, skipping training step")
        skip_training = input("Do you want to retrain? (y/N): ").strip().lower() == 'y'
    else:
        skip_training = False
    
    if not skip_training:
        success = run_command(
            [sys.executable, "train_dual_encoder.py"],
            "Training Dual Encoder Model"
        )
        if not success:
            print("\n❌ Training failed. Aborting workflow.")
            return 1
    
    # Step 2: Build FAISS index (if not already built)
    if check_faiss_index():
        print("\n✅ FAISS index already built")
        skip_index = input("Do you want to rebuild? (y/N): ").strip().lower() == 'y'
    else:
        skip_index = False
    
    if not skip_index:
        success = run_command(
            [sys.executable, "build_faiss_dual.py"],
            "Building FAISS Index"
        )
        if not success:
            print("\n❌ Index building failed. Aborting workflow.")
            return 1
    
    # Step 3: Test the system
    print("\n" + "="*60)
    print("TESTING THE SYSTEM")
    print("="*60)
    
    try:
        import torch
        import faiss
        from sentence_transformers import SentenceTransformer
        from dual_encoder_model import DualEncoder512
        
        # Load everything
        print("\nLoading system components...")
        
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {DEVICE}")
        
        text_encoder = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2",
            device=DEVICE
        )
        print("✅ Text encoder loaded")
        
        model = DualEncoder512().to(DEVICE)
        model.load_state_dict(torch.load("./models/dual_encoder_512.pt", map_location=DEVICE))
        model.eval()
        print("✅ Dual encoder loaded")
        
        index = faiss.read_index("./faiss/index.faiss")
        print(f"✅ FAISS index loaded ({index.ntotal} documents)")
        
        meta = json.load(open("./faiss/meta.json"))
        print("✅ Metadata loaded")
        
        # Test query
        test_query = "deep learning for document analysis"
        print(f"\nTest query: '{test_query}'")
        
        with torch.no_grad():
            q = text_encoder.encode([test_query], convert_to_tensor=True, device=DEVICE)
            q_emb = model.encode_query(q).cpu().numpy()
        
        scores, indices = index.search(q_emb, 5)
        
        print("\nTop 5 results:")
        for rank, idx in enumerate(indices[0], start=1):
            print(f"  {rank}. {meta[idx]['arxiv_id']} (score: {scores[0][rank-1]:.4f})")
        
        print("\n✅ System test passed!")
        
    except Exception as e:
        print(f"\n❌ System test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Final message
    print("\n" + "="*60)
    print("WORKFLOW COMPLETE")
    print("="*60)
    print("\n🎉 All steps completed successfully!")
    print("\n📋 Next steps:")
    print("   Launch the web interface:")
    print("   $ streamlit run frontend.py")
    print("\n   Or use the system programmatically by importing the modules")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())