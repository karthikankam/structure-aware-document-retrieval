"""
Test individual components of the document retrieval system
Useful for debugging and verification
"""

import json
import torch
import numpy as np
from pathlib import Path

def test_embeddings():
    """Test loading and format of embeddings"""
    print("\n" + "="*60)
    print("TESTING EMBEDDINGS")
    print("="*60)
    
    emb_dir = Path("./embeddings")
    if not emb_dir.exists():
        print("❌ Embeddings directory not found")
        return False
    
    emb_files = list(emb_dir.glob("*.json"))
    if not emb_files:
        print("❌ No embedding files found")
        return False
    
    print(f"✅ Found {len(emb_files)} embedding files")
    
    # Test first few files
    test_count = min(3, len(emb_files))
    for i, emb_file in enumerate(emb_files[:test_count], 1):
        print(f"\nFile {i}: {emb_file.name}")
        
        try:
            data = json.load(open(emb_file))
            
            # Check arxiv_id
            if "arxiv_id" in data:
                print(f"  ✅ arxiv_id: {data['arxiv_id']}")
            else:
                print(f"  ⚠️  No arxiv_id field")
            
            # Check embedding format
            emb = None
            if "text_embedding" in data:
                emb = data["text_embedding"]
                if isinstance(emb, list) and len(emb) > 0:
                    if isinstance(emb[0], list):
                        dim = len(emb[0])
                        print(f"  ✅ text_embedding (nested): {dim} dimensions")
                        emb = emb[0]
                    else:
                        dim = len(emb)
                        print(f"  ✅ text_embedding (flat): {dim} dimensions")
            elif "embedding" in data:
                emb = data["embedding"]
                dim = len(emb)
                print(f"  ✅ embedding: {dim} dimensions")
            else:
                print(f"  ❌ No embedding field found")
                continue
            
            # Check dimension
            if dim != 512:
                print(f"  ⚠️  Expected 512 dimensions, got {dim}")
            
            # Check values
            emb_array = np.array(emb)
            print(f"  Stats: min={emb_array.min():.4f}, max={emb_array.max():.4f}, mean={emb_array.mean():.4f}")
            
            # Check additional fields
            if "num_pages" in data:
                print(f"  ✅ num_pages: {data['num_pages']}")
            if "num_elements" in data:
                print(f"  ✅ num_elements: {data['num_elements']}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    return True

def test_metadata():
    """Test loading metadata"""
    print("\n" + "="*60)
    print("TESTING METADATA")
    print("="*60)
    
    meta_dir = Path("./metadata")
    if not meta_dir.exists():
        print("❌ Metadata directory not found")
        return False
    
    meta_files = list(meta_dir.glob("*.json"))
    if not meta_files:
        print("❌ No metadata files found")
        return False
    
    print(f"✅ Found {len(meta_files)} metadata files")
    
    # Test first few files
    test_count = min(3, len(meta_files))
    for i, meta_file in enumerate(meta_files[:test_count], 1):
        print(f"\nFile {i}: {meta_file.name}")
        
        try:
            data = json.load(open(meta_file))
            
            if "arxiv_id" in data:
                print(f"  ✅ arxiv_id: {data['arxiv_id']}")
            
            if "title" in data:
                title = data['title'][:60] + "..." if len(data['title']) > 60 else data['title']
                print(f"  ✅ title: {title}")
            
            if "abstract" in data:
                abstract = data['abstract'][:60] + "..." if len(data['abstract']) > 60 else data['abstract']
                print(f"  ✅ abstract: {abstract}")
            
            if "authors" in data:
                print(f"  ✅ authors: {len(data['authors'])} authors")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    return True

def test_dataset():
    """Test the DualEncoderDataset"""
    print("\n" + "="*60)
    print("TESTING DATASET")
    print("="*60)
    
    try:
        from dual_encoder_dataset import DualEncoderDataset
        
        dataset = DualEncoderDataset(
            embeddings_dir="./embeddings",
            metadata_dir="./metadata"
        )
        
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        # Test loading a sample
        if len(dataset) > 0:
            query, doc_emb = dataset[0]
            print(f"\nSample 0:")
            print(f"  Query length: {len(query)} chars")
            print(f"  Query preview: {query[:100]}...")
            print(f"  Document embedding shape: {doc_emb.shape}")
            print(f"  Document embedding dtype: {doc_emb.dtype}")
            
            # Check if embedding is normalized or raw
            norm = torch.norm(doc_emb).item()
            print(f"  Embedding norm: {norm:.4f}")
            if abs(norm - 1.0) < 0.01:
                print(f"  ✅ Embedding appears normalized")
            else:
                print(f"  ℹ️  Embedding is not normalized (will be normalized in model)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model():
    """Test the DualEncoder512 model"""
    print("\n" + "="*60)
    print("TESTING MODEL")
    print("="*60)
    
    try:
        from dual_encoder_model import DualEncoder512
        
        model = DualEncoder512()
        print(f"✅ Model created")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        batch_size = 4
        query_dim = 768
        doc_dim = 512
        
        query_emb = torch.randn(batch_size, query_dim)
        doc_emb = torch.randn(batch_size, doc_dim)
        
        with torch.no_grad():
            q_out = model.encode_query(query_emb)
            d_out = model.encode_doc(doc_emb)
        
        print(f"\nForward pass test:")
        print(f"  Query input: {query_emb.shape} -> Output: {q_out.shape}")
        print(f"  Doc input: {doc_emb.shape} -> Output: {d_out.shape}")
        
        # Check normalization
        q_norms = torch.norm(q_out, dim=1)
        d_norms = torch.norm(d_out, dim=1)
        print(f"  Query output norms: {q_norms.tolist()}")
        print(f"  Doc output norms: {d_norms.tolist()}")
        
        if torch.allclose(q_norms, torch.ones_like(q_norms), atol=1e-6):
            print(f"  ✅ Query outputs are normalized")
        if torch.allclose(d_norms, torch.ones_like(d_norms), atol=1e-6):
            print(f"  ✅ Doc outputs are normalized")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trained_model():
    """Test loading trained model"""
    print("\n" + "="*60)
    print("TESTING TRAINED MODEL")
    print("="*60)
    
    model_path = Path("./models/dual_encoder_512.pt")
    if not model_path.exists():
        print(f"⚠️  Trained model not found at {model_path}")
        print("   Run training first: python train_dual_encoder.py")
        return False
    
    try:
        from dual_encoder_model import DualEncoder512
        
        model = DualEncoder512()
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"✅ Loaded trained model from {model_path}")
        
        # Test with dummy data
        with torch.no_grad():
            query = torch.randn(1, 768)
            doc = torch.randn(1, 512)
            
            q_out = model.encode_query(query)
            d_out = model.encode_doc(doc)
            
            similarity = (q_out * d_out).sum().item()
            print(f"  Test similarity: {similarity:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading trained model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_faiss_index():
    """Test FAISS index"""
    print("\n" + "="*60)
    print("TESTING FAISS INDEX")
    print("="*60)
    
    index_path = Path("./faiss/index.faiss")
    meta_path = Path("./faiss/meta.json")
    
    if not index_path.exists():
        print(f"⚠️  FAISS index not found at {index_path}")
        print("   Build index first: python build_faiss_dual.py")
        return False
    
    try:
        import faiss
        
        index = faiss.read_index(str(index_path))
        print(f"✅ Loaded FAISS index")
        print(f"  Total documents: {index.ntotal}")
        print(f"  Dimension: {index.d}")
        print(f"  Index type: {type(index).__name__}")
        
        if meta_path.exists():
            meta = json.load(open(meta_path))
            print(f"✅ Loaded metadata for {len(meta)} documents")
            
            # Check consistency
            if len(meta) == index.ntotal:
                print(f"  ✅ Metadata count matches index")
            else:
                print(f"  ⚠️  Metadata count ({len(meta)}) doesn't match index ({index.ntotal})")
            
            # Show sample
            if meta:
                print(f"\n  Sample metadata:")
                sample = meta[0]
                for key, value in sample.items():
                    print(f"    {key}: {value}")
        
        # Test search
        query_emb = np.random.randn(1, index.d).astype('float32')
        query_emb = query_emb / np.linalg.norm(query_emb)  # Normalize
        
        scores, indices = index.search(query_emb, 5)
        print(f"\n  Test search results:")
        for i, idx in enumerate(indices[0][:5]):
            print(f"    {i+1}. Index {idx}, Score: {scores[0][i]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing FAISS index: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("COMPONENT TESTING SUITE")
    print("="*60)
    
    results = {}
    
    # Run tests
    results['embeddings'] = test_embeddings()
    results['metadata'] = test_metadata()
    results['dataset'] = test_dataset()
    results['model'] = test_model()
    results['trained_model'] = test_trained_model()
    results['faiss_index'] = test_faiss_index()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nPassed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()