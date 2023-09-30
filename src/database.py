from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

class DatabaseManager:
    def __init__(self, embeddings, db_path):
        self.embeddings = embeddings
        self.db_path = db_path

    def load_db(self):
        db = FAISS.load_local(self.db_path, self.embeddings)
        return db
