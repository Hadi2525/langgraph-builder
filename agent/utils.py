from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

def split_text(documents: list[Document],
               chunk_size: int = 1000,
               chunk_overlap: int = 200) -> list[Document]:
    """Splits the text into smaller chunks for better processing.
    args:
        documents (list[Document]): List of documents to be split.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.
    returns:
        list[Document]: List of split documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    return texts

def format_citations(citations: list[Document]) -> str:
    """Formats the citations for better readability.
    args:
        citations (list[Document]): List of citations to be formatted.
    returns:
        str: Formatted citations.
    """
    formatted_citations = []
    for citation, score in citations:
        citation_json = {
            "title": citation.metadata.get("title", ""),
            "page_label": citation.metadata.get("page_label", ""),
            "score": score,
            "page_content": citation.page_content
        }
        formatted_citations.append(citation_json)
    return json.dumps(formatted_citations, indent=2)
