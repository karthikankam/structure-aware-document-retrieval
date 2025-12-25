"""
ArXiv Data Collection Module for Multi-Modal Scientific Retrieval
Collects papers with metadata, PDFs, and extracts structural information
"""

import arxiv
import requests
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import time
from datetime import datetime
import PyPDF2
from io import BytesIO

class ArXivDataCollector:
    """Collects and processes ArXiv papers for document structure analysis"""
    
    def __init__(self, output_dir: str = "./arxiv_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.pdf_dir = self.output_dir / "pdfs"
        self.metadata_dir = self.output_dir / "metadata"
        self.processed_dir = self.output_dir / "processed"
        
        for dir_path in [self.pdf_dir, self.metadata_dir, self.processed_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def search_papers(self, 
                     query: str, 
                     max_results: int = 100,
                     categories: Optional[List[str]] = None) -> List[Dict]:
        """
        Search ArXiv papers using the API
        
        Args:
            query: Search query string
            max_results: Maximum number of papers to retrieve
            categories: Filter by ArXiv categories (e.g., ['cs.CV', 'cs.AI'])
        
        Returns:
            List of paper metadata dictionaries
        """
        print(f"Searching ArXiv for: {query}")
        
        # Build search query with categories
        if categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            full_query = f"({query}) AND ({cat_query})"
        else:
            full_query = query
        
        # Search ArXiv
        client = arxiv.Client()
        search = arxiv.Search(
            query=full_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers = []
        try:
            for result in client.results(search):
                paper_data = {
                    'arxiv_id': result.entry_id.split('/')[-1],
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'abstract': result.summary,
                    'categories': result.categories,
                    'published': result.published.isoformat(),
                    'updated': result.updated.isoformat(),
                    'pdf_url': result.pdf_url,
                    'comment': result.comment,
                    'journal_ref': result.journal_ref,
                    'primary_category': result.primary_category
                }
                papers.append(paper_data)
                print(f"Found: {paper_data['title'][:60]}...")
        except arxiv.UnexpectedEmptyPageError as e:
            print(f"Warning: ArXiv returned empty page for query '{query}'. Got {len(papers)} results so far.")
            print(f"This usually means the query has fewer results than requested.")
        except Exception as e:
            print(f"Error searching for '{query}': {e}")
            print(f"Got {len(papers)} results before error.")
        
        print(f"Total papers found for '{query}': {len(papers)}")
        return papers
    
    def download_pdf(self, paper_id: str, pdf_url: str) -> Optional[Path]:
        """Download PDF for a paper"""
        pdf_path = self.pdf_dir / f"{paper_id}.pdf"
        
        if pdf_path.exists():
            print(f"PDF already exists: {paper_id}")
            return pdf_path
        
        try:
            print(f"Downloading PDF: {paper_id}")
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            time.sleep(1)  # Rate limiting
            return pdf_path
            
        except Exception as e:
            print(f"Error downloading {paper_id}: {e}")
            return None
    
    def extract_pdf_metadata(self, pdf_path: Path) -> Dict:
        """Extract basic metadata from PDF"""
        try:
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                metadata = {
                    'num_pages': len(pdf_reader.pages),
                    'pdf_metadata': pdf_reader.metadata.__dict__ if pdf_reader.metadata else {}
                }
                
                # Extract text from first page for structure analysis
                if len(pdf_reader.pages) > 0:
                    first_page_text = pdf_reader.pages[0].extract_text()
                    metadata['first_page_preview'] = first_page_text[:500]
                
                return metadata
                
        except Exception as e:
            print(f"Error extracting PDF metadata: {e}")
            return {}
    
    def collect_dataset(self, 
                       queries: List[str], 
                       papers_per_query: int = 50,
                       download_pdfs: bool = True) -> Dict:
        """
        Collect a complete dataset
        
        Args:
            queries: List of search queries
            papers_per_query: Number of papers per query
            download_pdfs: Whether to download PDFs
        
        Returns:
            Dictionary with collection statistics
        """
        all_papers = []
        unique_papers = set()
        
        for query in queries:
            papers = self.search_papers(query, max_results=papers_per_query)
            
            for paper in papers:
                paper_id = paper['arxiv_id']
                
                # Avoid duplicates
                if paper_id in unique_papers:
                    continue
                    
                unique_papers.add(paper_id)
                
                # Save metadata
                metadata_path = self.metadata_dir / f"{paper_id}.json"
                with open(metadata_path, 'w') as f:
                    json.dump(paper, f, indent=2)
                
                # Download PDF
                if download_pdfs:
                    pdf_path = self.download_pdf(paper_id, paper['pdf_url'])
                    
                    if pdf_path:
                        # Extract PDF metadata
                        pdf_metadata = self.extract_pdf_metadata(pdf_path)
                        paper['pdf_metadata'] = pdf_metadata
                        
                        # Update metadata file
                        with open(metadata_path, 'w') as f:
                            json.dump(paper, f, indent=2)
                
                all_papers.append(paper)
        
        # Save collection summary
        summary = {
            'total_papers': len(all_papers),
            'queries': queries,
            'collection_date': datetime.now().isoformat(),
            'papers_per_query': papers_per_query
        }
        
        summary_path = self.output_dir / "collection_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n=== Collection Complete ===")
        print(f"Total papers collected: {len(all_papers)}")
        print(f"Data saved to: {self.output_dir}")
        
        return summary


# Example usage
if __name__ == "__main__":
    # Initialize collector
    collector = ArXivDataCollector(output_dir="./arxiv_retrieval_data")
    
    # Define search queries for diverse document structures
    queries = [
        "document layout analysis",
        "table detection recognition",
        "mathematical formula recognition",
        "figure caption extraction",
        "document structure understanding"
    ]
    
    # Collect dataset
    summary = collector.collect_dataset(
        queries=queries,
        papers_per_query=20,
        download_pdfs=True
    )
    
    print("\nDataset collection complete!")
    print(f"Summary: {summary}")