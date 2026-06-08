# Structure-Aware Document Retrieval

A neural document retrieval system for scientific papers. It maps natural-language
queries and documents into a shared 512-dimensional embedding space using a
**dual-encoder** trained with **contrastive learning**, and serves fast nearest-neighbor
search through a **FAISS** index with an interactive **Streamlit** frontend.

The pipeline is built around arXiv papers (computer vision, NLP, and machine learning
categories) and is designed to support structure-aware document understanding —
combining textual, visual, and layout features into a single representation.

checkout the paper on SMMESR.pdf
---

## Features

- **Dual-encoder retrieval** — separate query and document projection heads map into a
  shared 512-d space (`dual_encoder_model.py`).
- **Contrastive training** — symmetric InfoNCE-style loss with temperature scaling
  (`contrastive_loss.py`).
- **arXiv data collection** — searches and downloads papers and metadata by query and
  category (`arxiv_data_collector.py`).
- **Document structure extraction** — extracts titles, sections, paragraphs, equations,
  figures, tables, and captions from PDFs (`document_structure_extractor.py`).
- **Structure-aware embeddings** — attention-based fusion of text, visual, and structural
  features (`structure_aware_embeddings.py`).
- **FAISS indexing** — inner-product index over L2-normalized vectors for cosine
  similarity search (`build_faiss_dual.py`).
- **Interactive search UI** — a Streamlit app for querying the index (`frontend.py`).
- **End-to-end orchestration** — one script that runs training, indexing, and checks
  (`complete_workflow.py`).

---

## Architecture

```
arXiv papers ──> Structure extraction ──> Structure-aware embeddings (512-d)
                                                       │
                                                       ▼
Query text ──> all-mpnet-base-v2 (768-d) ──> Query encoder ─┐
                                                            ├─> shared 512-d space
Document embedding (512-d) ──────────────> Document encoder ┘
                                                       │
                                                       ▼
                                              FAISS index (cosine)
                                                       │
                                                       ▼
                                          Streamlit search frontend
```

- **Query side:** text is embedded by the frozen `sentence-transformers/all-mpnet-base-v2`
  model (768-d), then projected to 512-d by a trainable query head.
- **Document side:** pre-computed 512-d document embeddings are refined by a trainable
  document head.
- Both outputs are L2-normalized, so inner product equals cosine similarity.

---

## Repository structure

| File / folder | Purpose |
|---|---|
| `arxiv_data_collector.py` | Collect papers, metadata, and PDFs from the arXiv API. |
| `document_structure_extractor.py` | Convert PDFs to images and extract structural elements. |
| `structure_aware_embeddings.py` | Fuse text/visual/structural features into embeddings. |
| `convert_embeddings.py` | Embedding format conversion helper (placeholder). |
| `dual_encoder_model.py` | `DualEncoder512` model (query + document projection heads). |
| `dual_encoder_dataset.py` | `Dataset` that pairs query text with document embeddings. |
| `contrastive_loss.py` | Symmetric contrastive loss with temperature. |
| `train_dual_encoder.py` | Train the dual encoder and save weights/config. |
| `build_faiss_dual.py` | Encode documents and build the FAISS index. |
| `frontend.py` | Streamlit search interface. |
| `complete_workflow.py` | Run the full pipeline (train → index → verify). |
| `test_components.py` | Component-level tests. |
| `verify_setup.py` | Environment and setup verification. |
| `config.json` | Central configuration (queries, model params, paths). |
| `models/` | Trained model (`dual_encoder_512.pt`) and config. |
| `faiss/` | FAISS index (`index.faiss`) and metadata (`meta.json`). |
| `dual_encoder_512_final.pt` | Final trained checkpoint. |
| `requirements.txt` | Python dependencies. |
| `LICENSE` | Apache-2.0 license. |

---

## Requirements

- Python 3.8+
- PyTorch 2.0+ (CUDA 11.8+ recommended for GPU; CPU also works)
- 16 GB+ RAM (32 GB recommended)
- ~50 GB disk space for the full dataset
- `poppler-utils` (required by `pdf2image` for PDF processing)

Key Python packages: `torch`, `transformers`, `sentence-transformers`, `faiss-cpu`
(or `faiss-gpu`), `arxiv`, `pdf2image`, `PyPDF2`, `opencv-python`, `numpy`, `pandas`,
`scikit-learn`, `streamlit`. See `requirements.txt` for the full list.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/karthikankam/structure-aware-document-retrieval.git
cd structure-aware-document-retrieval

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install PyTorch (pick the build for your CUDA version from pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install the remaining dependencies
pip install -r requirements.txt

# 5. Install system dependencies (poppler)
#    Ubuntu/Debian:  sudo apt-get install poppler-utils
#    macOS:          brew install poppler
#    Windows:        https://github.com/oschwartz10612/poppler-windows/releases/
```

Verify the install:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python verify_setup.py
```

---

## Configuration

All key settings live in `config.json`:

- **`collection.queries`** — arXiv search queries used for data collection.
- **`collection.papers_per_query`** — papers to fetch per query (default 100).
- **`collection.categories`** — arXiv categories (`cs.CV`, `cs.CL`, `cs.LG`).
- **`model`** — text model (`all-mpnet-base-v2`), embedding dims (768 → 512),
  batch size, epochs, learning rate, temperature, and device.
- **`paths`** — directories for embeddings, metadata, FAISS index, models, and structures.

---

## Usage

### Option A — run the full pipeline

```bash
python complete_workflow.py
```

This checks prerequisites (embeddings and metadata directories), trains the dual encoder,
and builds the FAISS index. When it finishes, launch the frontend as shown below.

### Option B — run each step manually

```bash
# 1. Collect arXiv papers and metadata
python arxiv_data_collector.py

# 2. Extract document structure from PDFs
python document_structure_extractor.py

# 3. Generate structure-aware embeddings
python structure_aware_embeddings.py

# 4. Train the dual encoder (saves models/dual_encoder_512.pt)
python train_dual_encoder.py

# 5. Build the FAISS index (saves faiss/index.faiss + faiss/meta.json)
python build_faiss_dual.py
```

### Launch the search interface

```bash
streamlit run frontend.py
```

Then open the local URL Streamlit prints, enter a query (e.g. *"transformer models for
document analysis"*), and adjust the number of results, similarity scores, and metadata
display from the sidebar.

---

## How it works

1. **Data collection** — `ArXivDataCollector` queries the arXiv API per configured query
   and category, then stores PDFs and metadata.
2. **Structure extraction** — PDFs are rasterized and parsed into typed `DocumentElement`s
   (title, section, paragraph, equation, figure, table, caption) with bounding boxes,
   page numbers, and confidence.
3. **Embedding** — `StructureAwareFusion` encodes each modality and fuses them via
   multi-head attention into a 512-d document vector.
4. **Training** — `DualEncoder512` is trained with a symmetric contrastive loss
   (temperature 0.05, AdamW, gradient clipping). The text encoder is frozen; only the
   projection heads are learned.
5. **Indexing** — documents are encoded and stored in a `faiss.IndexFlatIP` index; vectors
   are L2-normalized so inner product gives cosine similarity.
6. **Retrieval** — at query time, text is embedded, projected, normalized, and matched
   against the index to return the top-k papers.

---

## Testing

```bash
python test_components.py
```

---

## License

Released under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [arXiv](https://arxiv.org/) for open access to scientific papers.
- [Sentence-Transformers](https://www.sbert.net/) (`all-mpnet-base-v2`).
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search.
- [Streamlit](https://streamlit.io/) for the frontend.
