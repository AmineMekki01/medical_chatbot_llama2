from langchain import PromptTemplate



class PromptTemplateManager:
    def __init__(self, template):
        self.template = template

    def set_custom_prompt(self):
        """
        Set the custom prompt
        
        Returns:    
        --------
            PromptTemplate: The custom prompt
        """
        prompt = PromptTemplate(template=self.template, input_variables=["context", "question"])
        return prompt