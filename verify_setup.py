"""
Verification script to check your document retrieval setup
Run this first to ensure everything is configured correctly
"""

import json
from pathlib import Path
import sys

def check_directory(path, name, required=True):
    """Check if directory exists and has files"""
    p = Path(path)
    exists = p.exists()
    is_dir = p.is_dir() if exists else False
    count = len(list(p.glob("*"))) if is_dir else 0
    
    status = "✅" if exists and is_dir else ("❌" if required else "⚠️")
    print(f"{status} {name}: {path}")
    if is_dir:
        print(f"   Files: {count}")
    elif required:
        print(f"   ERROR: Directory not found!")
    else:
        print(f"   Not found (optional)")
    
    return exists and is_dir, count

def check_embedding_format(emb_path):
    """Check the format of an embedding file"""
    try:
        with open(emb_path, 'r') as f:
            data = json.load(f)
        
        print(f"\n📄 Sample embedding file: {emb_path.name}")
        print(f"   Keys: {list(data.keys())}")
        
        if "text_embedding" in data:
            emb = data["text_embedding"]
            if isinstance(emb, list) and len(emb) > 0:
                dim = len(emb[0]) if isinstance(emb[0], list) else len(emb)
                print(f"   Format: text_embedding (nested list)")
                print(f"   Dimension: {dim}")
            else:
                print(f"   Format: text_embedding (unexpected structure)")
        elif "embedding" in data:
            emb = data["embedding"]
            dim = len(emb)
            print(f"   Format: embedding (flat list)")
            print(f"   Dimension: {dim}")
        else:
            print(f"   ⚠️  No 'embedding' or 'text_embedding' key found!")
            return False
        
        if "arxiv_id" in data:
            print(f"   ArXiv ID: {data['arxiv_id']}")
        if "num_pages" in data:
            print(f"   Pages: {data['num_pages']}")
        if "num_elements" in data:
            print(f"   Elements: {data['num_elements']}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return False

def main():
    print("="*60)
    print("DOCUMENT RETRIEVAL SYSTEM - SETUP VERIFICATION")
    print("="*60)
    
    # Check directories
    print("\n📁 Checking directories...")
    
    emb_exists, emb_count = check_directory("./embeddings", "Embeddings", required=True)
    meta_exists, meta_count = check_directory("./metadata", "Metadata", required=True)
    check_directory("./models", "Models", required=False)
    check_directory("./faiss", "FAISS Index", required=False)
    check_directory("./structures", "Structures", required=False)
    
    # Check embedding format
    if emb_exists and emb_count > 0:
        emb_files = list(Path("./embeddings").glob("*.json"))
        if emb_files:
            check_embedding_format(emb_files[0])
    
    # Check metadata format
    if meta_exists and meta_count > 0:
        print("\n📄 Checking metadata format...")
        meta_files = list(Path("./metadata").glob("*.json"))
        if meta_files:
            try:
                with open(meta_files[0], 'r') as f:
                    meta = json.load(f)
                print(f"   Sample file: {meta_files[0].name}")
                print(f"   Keys: {list(meta.keys())}")
                if "title" in meta:
                    print(f"   Title: {meta['title'][:60]}...")
                if "abstract" in meta:
                    print(f"   Abstract: {meta['abstract'][:60]}...")
            except Exception as e:
                print(f"   ❌ Error reading metadata: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    issues = []
    
    if not emb_exists:
        issues.append("Embeddings directory missing")
    elif emb_count == 0:
        issues.append("No embedding files found")
    
    if not meta_exists:
        issues.append("Metadata directory missing")
    elif meta_count == 0:
        issues.append("No metadata files found")
    
    if not Path("./models").exists():
        issues.append("Models directory not created yet (will be created during training)")
    
    if not Path("./faiss").exists():
        issues.append("FAISS directory not created yet (will be created when building index)")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n📋 Next steps:")
        print("   1. Ensure embeddings and metadata directories exist")
        print("   2. Run: python train_dual_encoder.py")
        print("   3. Run: python build_faiss_dual.py")
        print("   4. Run: streamlit run frontend.py")
    else:
        print("\n✅ Setup looks good!")
        print("\n📋 Next steps:")
        if not Path("./models/dual_encoder_512.pt").exists():
            print("   1. Train model: python train_dual_encoder.py")
        if not Path("./faiss/index.faiss").exists():
            print("   2. Build index: python build_faiss_dual.py")
        print("   3. Launch app: streamlit run frontend.py")
    
    print("="*60)

if __name__ == "__main__":
    main()