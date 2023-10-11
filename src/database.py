from langchain.vectorstores import FAISS

class DatabaseManager:
    def __init__(self, embeddings, db_path):
        self.embeddings = embeddings
        self.db_path = db_path

    def load_db(self):
        """
        Load the database from the db_path
        
        Returns:   
            FAISS: The FAISS database        
        """
        db = FAISS.load_local(self.db_path, self.embeddings)
        return db
