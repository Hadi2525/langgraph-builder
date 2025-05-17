from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils import split_text

load_dotenv()

VECTORSTORE_PATH = "./vectorstore"

class VectorStore:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-large")
    
    def upload_documents(self):
        """Uploads documents to the vector store."""
        
        try:
            os.makedirs(VECTORSTORE_PATH, exist_ok=True)
            logger.info(f"Vector store path: {VECTORSTORE_PATH}")
            document_loader = PyPDFDirectoryLoader(
                self.pdf_path)
            
            documents = document_loader.load()
            texts = split_text(documents)

            db = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=VECTORSTORE_PATH,   
            )

            db.persist()
            logger.info("Documents uploaded to vector store.")
            logger.info(f"Vector store path: {VECTORSTORE_PATH}")
            logger.info(f"Number of documents in vector store: {len(db)}")
            logger.info(f"Number of chunks in vector store: {len(texts)}")

        except Exception as e:
            logger.error(f"Error uploading documents: {e}")
            raise e
        


