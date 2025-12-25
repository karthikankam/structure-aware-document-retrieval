"""
Document Structure Extraction Module
Extracts structural elements from scientific PDFs using vision and layout analysis
"""

import torch
import torch.nn as nn
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pdf2image
from dataclasses import dataclass, asdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DocumentElement:
    """Represents a structural element in a document"""
    type: str  # 'title', 'section', 'paragraph', 'equation', 'figure', 'table', 'caption'
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    page: int
    confidence: float
    text: str = ""
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DocumentStructure:
    """Complete structural representation of a document"""
    arxiv_id: str
    num_pages: int
    elements: List[DocumentElement]
    layout_graph: Dict  # Relationships between elements
    features: Dict  # Global document features


class DocumentStructureExtractor:
    """Extracts structural information from scientific PDFs"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize models (using lightweight alternatives for demonstration)
        # In production, use LayoutLMv3, Detectron2, or specialized models
        self.processor = None
        self.layout_model = None
        
    def pdf_to_images(self, pdf_path: Path, dpi: int = 150) -> List[Image.Image]:
        """Convert PDF pages to images"""
        try:
            # Try with pdf2image first
            images = pdf2image.convert_from_path(
                str(pdf_path), 
                dpi=dpi,
                fmt='jpeg',
                thread_count=2
            )
            return images
        except Exception as e:
            if "poppler" in str(e).lower():
                # Try alternative method using PyMuPDF
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(str(pdf_path))
                    images = []
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        # Render page to image
                        mat = fitz.Matrix(dpi/72, dpi/72)  # Create zoom matrix
                        pix = page.get_pixmap(matrix=mat)
                        # Convert to PIL Image
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        images.append(img)
                    doc.close()
                    return images
                except ImportError:
                    print(f"Error: Neither poppler nor PyMuPDF is available")
                    print(f"Install with: pip install PyMuPDF")
                    return []
                except Exception as e2:
                    print(f"Error with PyMuPDF: {e2}")
                    return []
            else:
                print(f"Error converting PDF to images: {e}")
                return []
    
    def detect_layout_elements(self, image: Image.Image, page_num: int) -> List[DocumentElement]:
        """
        Detect layout elements using computer vision
        This is a simplified version - in production use models like:
        - LayoutLMv3
        - Detectron2 with layout detection
        - DocLayout-YOLO
        """
        elements = []
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Simple heuristic-based detection (replace with deep learning model)
        # Detect text regions using contours
        try:
            # Apply thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Limit number of contours to process
            max_contours = min(100, len(contours))
            
            for idx, contour in enumerate(contours[:max_contours]):
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter small contours
                if w < 50 or h < 20:
                    continue
                
                # Filter very large contours (likely page borders)
                if w > image.width * 0.95 or h > image.height * 0.95:
                    continue
                
                # Classify based on heuristics (replace with ML classifier)
                element_type = self._classify_element(w, h, y, image.height, image.width)
                
                element = DocumentElement(
                    type=element_type,
                    bbox=(x, y, x+w, y+h),
                    page=page_num,
                    confidence=0.8,
                    text="",
                    metadata={'area': w*h, 'aspect_ratio': w/h if h > 0 else 0}
                )
                elements.append(element)
                
        except Exception as e:
            print(f"Error detecting elements on page {page_num}: {e}")
        
        # Remove duplicate/overlapping elements
        elements = self._remove_overlaps(elements)
        
        return elements
    
    def _classify_element(self, w: int, h: int, y: int, page_height: int, page_width: int) -> str:
        """Simple heuristic classification - replace with ML model"""
        aspect_ratio = w / h if h > 0 else 0
        relative_y = y / page_height if page_height > 0 else 0
        relative_w = w / page_width if page_width > 0 else 0
        
        # Title usually at top, wide
        if relative_y < 0.15 and aspect_ratio > 3 and relative_w > 0.5:
            return 'title'
        
        # Section headers are wide but not too tall
        elif relative_w > 0.6 and aspect_ratio > 5:
            return 'section'
        
        # Equations are usually centered with specific aspect ratio
        elif 0.5 < aspect_ratio < 5 and h < 150 and relative_w < 0.7:
            return 'equation'
        
        # Tables are wider with multiple rows
        elif aspect_ratio > 2 and h < 400 and h > 100:
            return 'table'
        
        # Figures are more square and larger
        elif 0.5 < aspect_ratio < 2 and w > 200 and h > 200:
            return 'figure'
        
        # Captions are narrow and below figures
        elif aspect_ratio > 4 and h < 80:
            return 'caption'
        
        # Default to paragraph
        else:
            return 'paragraph'
    
    def _remove_overlaps(self, elements: List[DocumentElement], 
                        iou_threshold: float = 0.5) -> List[DocumentElement]:
        """Remove overlapping elements based on IoU"""
        if len(elements) <= 1:
            return elements
        
        # Sort by area (keep larger elements)
        elements = sorted(elements, 
                         key=lambda e: (e.bbox[2]-e.bbox[0])*(e.bbox[3]-e.bbox[1]), 
                         reverse=True)
        
        keep = []
        for elem in elements:
            should_keep = True
            for kept_elem in keep:
                iou = self._compute_iou(elem.bbox, kept_elem.bbox)
                if iou > iou_threshold:
                    should_keep = False
                    break
            if should_keep:
                keep.append(elem)
        
        return keep
    
    def _compute_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Compute Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def build_layout_graph(self, elements: List[DocumentElement]) -> Dict:
        """
        Build a graph representing spatial and semantic relationships
        """
        graph = {
            'nodes': [],
            'edges': []
        }
        
        # Add nodes
        for idx, elem in enumerate(elements):
            graph['nodes'].append({
                'id': idx,
                'type': elem.type,
                'page': elem.page,
                'bbox': elem.bbox
            })
        
        # Add edges based on spatial relationships
        for i, elem1 in enumerate(elements):
            for j, elem2 in enumerate(elements):
                if i >= j:
                    continue
                
                # Same page relationships
                if elem1.page == elem2.page:
                    relation = self._get_spatial_relation(elem1.bbox, elem2.bbox)
                    if relation:
                        graph['edges'].append({
                            'source': i,
                            'target': j,
                            'relation': relation
                        })
                
                # Sequential pages
                elif elem2.page == elem1.page + 1:
                    graph['edges'].append({
                        'source': i,
                        'target': j,
                        'relation': 'next_page'
                    })
        
        return graph
    
    def _get_spatial_relation(self, bbox1: Tuple, bbox2: Tuple) -> str:
        """Determine spatial relationship between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        center_y1 = (y1_1 + y2_1) / 2
        center_y2 = (y1_2 + y2_2) / 2
        
        center_x1 = (x1_1 + x2_1) / 2
        center_x2 = (x1_2 + x2_2) / 2
        
        # Vertical relationship
        if abs(center_y1 - center_y2) < 50:
            if center_x1 < center_x2:
                return 'left_of'
            else:
                return 'right_of'
        elif center_y1 < center_y2:
            return 'above'
        else:
            return 'below'
    
    def extract_global_features(self, elements: List[DocumentElement], 
                               num_pages: int) -> Dict:
        """Extract document-level features"""
        features = {
            'num_pages': num_pages,
            'total_elements': len(elements),
            'element_types': {},
            'avg_elements_per_page': len(elements) / num_pages if num_pages > 0 else 0,
            'has_equations': False,
            'has_figures': False,
            'has_tables': False,
            'layout_density': 0.0
        }
        
        # Count element types
        for elem in elements:
            features['element_types'][elem.type] = \
                features['element_types'].get(elem.type, 0) + 1
            
            if elem.type == 'equation':
                features['has_equations'] = True
            elif elem.type == 'figure':
                features['has_figures'] = True
            elif elem.type == 'table':
                features['has_tables'] = True
        
        # Calculate layout density (elements per unit area)
        if elements:
            total_area = sum(
                (e.bbox[2] - e.bbox[0]) * (e.bbox[3] - e.bbox[1]) 
                for e in elements
            )
            # Normalize by typical page size (assuming letter size at 150 DPI)
            page_area = (1275 * 1650) * num_pages  # Approximate
            features['layout_density'] = total_area / page_area if page_area > 0 else 0
        
        return features
    
    def process_document(self, pdf_path: Path, arxiv_id: str) -> Optional[DocumentStructure]:
        """Process a complete document and extract structure"""
        print(f"Processing document: {arxiv_id}")
        
        # Convert PDF to images
        images = self.pdf_to_images(pdf_path)
        
        if not images:
            print(f"Failed to process {arxiv_id}")
            return None
        
        # Extract elements from each page
        all_elements = []
        for page_num, image in enumerate(images):
            try:
                elements = self.detect_layout_elements(image, page_num)
                all_elements.extend(elements)
            except Exception as e:
                print(f"Error processing page {page_num} of {arxiv_id}: {e}")
                continue
        
        if not all_elements:
            print(f"No elements extracted from {arxiv_id}")
            # Create minimal structure
            all_elements = [DocumentElement(
                type='paragraph',
                bbox=(0, 0, 100, 100),
                page=0,
                confidence=0.5,
                text="",
                metadata={}
            )]
        
        # Build layout graph
        layout_graph = self.build_layout_graph(all_elements)
        
        # Extract global features
        global_features = self.extract_global_features(all_elements, len(images))
        
        # Create document structure
        doc_structure = DocumentStructure(
            arxiv_id=arxiv_id,
            num_pages=len(images),
            elements=all_elements,
            layout_graph=layout_graph,
            features=global_features
        )
        
        return doc_structure
    
    def save_structure(self, doc_structure: DocumentStructure, output_path: Path):
        """Save document structure to JSON"""
        # Convert to serializable format
        data = {
            'arxiv_id': doc_structure.arxiv_id,
            'num_pages': doc_structure.num_pages,
            'elements': [asdict(elem) for elem in doc_structure.elements],
            'layout_graph': doc_structure.layout_graph,
            'features': doc_structure.features
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved structure to {output_path}")
    
    def load_structure(self, structure_path: Path) -> DocumentStructure:
        """Load document structure from JSON"""
        with open(structure_path, 'r') as f:
            data = json.load(f)
        
        # Convert back to DocumentElement objects
        elements = [
            DocumentElement(**elem) for elem in data['elements']
        ]
        
        doc_structure = DocumentStructure(
            arxiv_id=data['arxiv_id'],
            num_pages=data['num_pages'],
            elements=elements,
            layout_graph=data['layout_graph'],
            features=data['features']
        )
        
        return doc_structure


# Example usage and batch processing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Document Structure Extractor')
    parser.add_argument('--pdf_dir', type=str, default='./arxiv_retrieval_data/pdfs',
                       help='Directory containing PDF files')
    parser.add_argument('--output_dir', type=str, default='./arxiv_retrieval_data/processed',
                       help='Output directory for structures')
    parser.add_argument('--single_file', type=str, default=None,
                       help='Process a single PDF file')
    parser.add_argument('--dpi', type=int, default=150,
                       help='DPI for PDF rendering')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = DocumentStructureExtractor()
    
    if args.single_file:
        # Process single file
        pdf_path = Path(args.single_file)
        if not pdf_path.exists():
            print(f"Error: File not found: {pdf_path}")
            exit(1)
        
        arxiv_id = pdf_path.stem
        output_path = Path(args.output_dir) / f"{arxiv_id}_structure.json"
        
        doc_structure = extractor.process_document(pdf_path, arxiv_id)
        
        if doc_structure:
            extractor.save_structure(doc_structure, output_path)
            print(f"\nDocument Structure Summary:")
            print(f"Pages: {doc_structure.num_pages}")
            print(f"Elements: {len(doc_structure.elements)}")
            print(f"Features: {doc_structure.features}")
        else:
            print("Failed to process document")
    
    else:
        # Batch process all PDFs
        pdf_dir = Path(args.pdf_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")
        
        if not pdf_files:
            print(f"No PDF files found in {pdf_dir}")
            exit(1)
        
        success_count = 0
        error_count = 0
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            arxiv_id = pdf_file.stem
            output_path = output_dir / f"{arxiv_id}_structure.json"
            
            # Skip if already processed
            if output_path.exists():
                print(f"Skipping {arxiv_id} (already processed)")
                success_count += 1
                continue
            
            try:
                doc_structure = extractor.process_document(pdf_file, arxiv_id)
                
                if doc_structure:
                    extractor.save_structure(doc_structure, output_path)
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                print(f"Error processing {arxiv_id}: {e}")
                error_count += 1
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Success: {success_count}/{len(pdf_files)}")
        print(f"Errors: {error_count}/{len(pdf_files)}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")