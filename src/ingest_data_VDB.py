from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"

def load_documents(data_path, glob_pattern="*.pdf", loader_cls=PyPDFLoader):
    loader = DirectoryLoader(data_path, glob=glob_pattern, loader_cls=loader_cls)
    return loader.load()

def split_text(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def create_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
    return embeddings

def create_vector_db(data_path=DATA_PATH, db_faiss_path=DB_FAISS_PATH):
    documents = load_documents(data_path)
    texts = split_text(documents)
    embeddings = create_embeddings(texts)
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(db_faiss_path)

    
if __name__ == "__main__":
    create_vector_db()
    
