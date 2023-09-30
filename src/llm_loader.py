from langchain.llms import CTransformers

class LLMLoader:
    @staticmethod
    def load_llm():
        llm = CTransformers(
            model="llama-2-7b-chat.ggmlv3.q2_K.bin",
            model_type="llama",
            max_new_tokens=512,
            temperature=0.5,
        )
        return llm
