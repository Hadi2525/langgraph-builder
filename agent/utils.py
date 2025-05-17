from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
