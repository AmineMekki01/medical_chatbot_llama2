from langchain.llms import CTransformers

class LLMLoader:
    @staticmethod
    def load_llm():
        """
        Load the LLM model  

        Returns:    
        --------
            LLM: The LLM model
        """
        llm = CTransformers(
            model="llama-2-7b-chat.ggmlv3.q2_K.bin",
            model_type="llama",
            max_new_tokens=512,
            temperature=0.5,
        )
        return llm
