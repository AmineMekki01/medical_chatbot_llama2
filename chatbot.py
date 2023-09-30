from src.prompt_template import PromptTemplateManager
from src.llm_loader import LLMLoader
from src.retrieval_qa import RetrievalQAChain
from src.database import DatabaseManager
from src.helpers import load_config
from langchain.embeddings import HuggingFaceEmbeddings

import chainlit as cl

config = load_config('config.yaml')

DB_FAISS_PATH = config['DB_FAISS_PATH']
custom_prompt_template = config['custom_prompt_template']


class QABot:
    def __init__(self, db_path):
        self.db_path = DB_FAISS_PATH

    def qa_bot(self):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
        db_manager = DatabaseManager(embeddings, self.db_path)
        db = db_manager.load_db()
        llm_loader = LLMLoader()
        llm = llm_loader.load_llm()
        prompt_manager = PromptTemplateManager(custom_prompt_template)
        qa_prompt = prompt_manager.set_custom_prompt()
        qa_chain_creator = RetrievalQAChain(llm, qa_prompt, db)
        qa = qa_chain_creator.create_qa_chain()
        return qa

def final_result():
    qa_bot_instance = QABot(DB_FAISS_PATH)
    qa = qa_bot_instance.qa_bot()
    qa_bot = cl.Bot.from_chain(qa)
    return qa_bot

## chainlit 
@cl.on_chat_start
async def start():
    qa_bot_instance = QABot(DB_FAISS_PATH)
    chain = qa_bot_instance.qa_bot()
    message = cl.Message(content="Starting the medical chatbot ..... ")
    await message.send()
    
    message.content = "Hi, Welcome to your Medical ChatBot. What is your request ? : "
    await message.update()
    cl.user_session.set("chain", chain)
    
@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]
    
    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += f"\nNo Sources Found"
        
    await cl.Message(content=answer).send()

