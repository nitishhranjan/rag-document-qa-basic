# src/document_loader.py
"""Document loading utilities."""

from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader


def process_all_pdfs(pdf_directory: str) -> List[Any]:
    """
    Process all PDF files in a directory.
    
    Args:
        pdf_directory: Path to directory containing PDFs
        
    Returns:
        List of loaded documents with metadata
    """
    all_documents = []
    pdf_directory = Path(pdf_directory)
    
    pdf_files = list(pdf_directory.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        print(f"\nProcessing {pdf_file.name}...")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            
            # Add source info to metadata
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'
            
            all_documents.extend(documents)
            print(f"Loaded {len(documents)} documents from {pdf_file}")
            
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
    
    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents