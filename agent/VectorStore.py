from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils import split_text

load_dotenv()

VECTORSTORE_PATH = "./vectorstore"

class VectorStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-large")
        
        data_path = Path(__file__).parent.parent / "data" / "pdf"
        vectorstore_dir = Path(__file__).parent.parent / "vectorstore"
        chroma_file = vectorstore_dir / "chroma.sqlite3"
        uuid_folders = [f for f in vectorstore_dir.iterdir() if f.is_dir() and len(f.name) == 36]
        if vectorstore_dir.exists() and chroma_file.exists() and uuid_folders:
            logger.info("Vectorstore already exists. Skipping document upload.")
            self.load_vectorstore()
            logger.info(f"Vector store path: {VECTORSTORE_PATH}")
            logger.info("Local vector store loaded.")
        else:
            self.upload_documents(data_path)

    def load_vectorstore(self):
        """Loads the vector store."""
        
        try:
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=VECTORSTORE_PATH,
            )
            logger.info(f"Vector store loaded from {VECTORSTORE_PATH}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise e
    
    def upload_documents(self, pdf_path: str):
        """Uploads documents to the vector store."""
        
        try:
            os.makedirs(VECTORSTORE_PATH, exist_ok=True)
            logger.info(f"Vector store path: {VECTORSTORE_PATH}")
            document_loader = PyPDFDirectoryLoader(pdf_path)
            
            documents = document_loader.load()
            texts = split_text(documents)

            self.vector_store = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=VECTORSTORE_PATH,   
            )

            self.vector_store.persist()
            logger.info("Documents uploaded to vector store.")

        except Exception as e:
            logger.error(f"Error uploading documents: {e}")
            raise e
        
        #TODO: Add other CRUD methods for vector store
    def search(self, query: str, search_type: str = "similarity"):
        """Searches for documents in the vector store."""
        
        try:
            if search_type == "similarity":
                results = self.vector_store.similarity_search(
                    query=query,
                    k=4,
                )


            elif search_type == "vector":
                results = self.vector_store.similarity_search_by_vector_with_relevance_scores(
                    embedding=self.embeddings.embed_query(query),
                    k=4
                )
            #TODO: results must be formatted to acconut for the langgraph state flow
            return results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise e




