from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"

def load_documents(data_path : str, glob_pattern : str ="*.pdf", loader_cls : PyPDFLoader = PyPDFLoader):
    """
    Load documents from a directory using a loader class.
    
    Parameters
    ----------
    data_path : str
        Path to directory containing documents.
    glob_pattern : str
        Glob pattern for matching documents.
    loader_cls : class
        Class for loading documents.
    
    Returns
    -------
    documents : list
        List of documents.  
    """
    loader = DirectoryLoader(data_path, glob=glob_pattern, loader_cls=loader_cls)
    return loader.load()

def split_text(documents : list, chunk_size : int = 500, chunk_overlap : int = 50):
    """
    Split documents into chunks of text.
    
    Parameters
    ----------
    documents : list
        List of documents.
    chunk_size : int
        Size of text chunks.
    chunk_overlap : int 
        Overlap between text chunks.
    
    Returns
    -------
    texts : list
        List of text chunks.    
        
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def create_embeddings(texts : list, model_name : str = "sentence-transformers/all-MiniLM-L6-v2", device : str = "cpu"):
    """
    Create embeddings from text chunks.
    
    Parameters
    ----------  
    texts : list
        List of text chunks.    
    model_name : str    
        Name of model to use for embeddings.
    device : str
        Device to use for embeddings.
        
    Returns 
    ------- 
    embeddings : class
        Embeddings class.
    
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
    return embeddings

def create_vector_db(data_path : str = DATA_PATH, db_faiss_path : str = DB_FAISS_PATH):
    """
    Create vector database from documents.
    
    Parameters  
    ----------
    data_path : str
        Path to directory containing documents. 
        
    db_faiss_path : str 
        Path to vector database.
    
    Returns 
    -------
    db : class
        Vector database.
    
    """
    
    documents = load_documents(data_path)
    texts = split_text(documents)
    embeddings = create_embeddings(texts)
    
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(db_faiss_path)

    
if __name__ == "__main__":
    create_vector_db()
    
