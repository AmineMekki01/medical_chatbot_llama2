from langchain.chains import RetrievalQA

class RetrievalQAChain:
    def __init__(self, llm, prompt, db):
        self.llm = llm # LLM model
        self.prompt = prompt # Prompt template 
        self.db = db   # VDB Faiss

    def create_qa_chain(self):
        """
        Create the QA chain
        
        Returns:    
        --------
            RetrievalQA: The QA chain   
            
        """
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
        return qa_chain
