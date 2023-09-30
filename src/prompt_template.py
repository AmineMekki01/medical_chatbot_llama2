from langchain import PromptTemplate



class PromptTemplateManager:
    def __init__(self, template):
        self.template = custom_prompt_template

    def set_custom_prompt(self):
        prompt = PromptTemplate(template=self.template, input_variables=["context", "question"])
        return prompt