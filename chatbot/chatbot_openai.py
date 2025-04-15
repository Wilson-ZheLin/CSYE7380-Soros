import logging
import os
import yaml
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class ChatbotOpenAI:

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s  %(message)s')

    def __init__(self, model_name="gpt-4o-mini-2024-07-18", api_key=os.getenv("OPENAI_API_KEY")):
        if api_key is None:
            raise ValueError("OpenAI API key is not set. Please provide it via OPENAI_API_KEY environment variable or constructor parameter")
        
        template_path = os.path.join(os.path.dirname(__file__), 'yml', 'prompt_template.yaml')
        with open(template_path, 'r') as file:
            self.templates = yaml.safe_load(file)
        self.model_name = model_name
        self.api_key = api_key

    def answer(self, context: str, query: str, chat_history=None):
        if chat_history is None:
            chat_history = []
            
        llm = ChatOpenAI(
            model_name=self.model_name,
            openai_api_key=self.api_key,
            temperature=0.1,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        full_prompt = []
        template = self.templates["chat_template"]
        system_content = "\n".join(template.split("\n")) + f"\nContext: {context}"
        full_prompt.append(SystemMessage(content=system_content))

        if chat_history:
            for message in chat_history:
                if message["role"] == "user":
                    full_prompt.append(HumanMessage(content=message["content"]))
                elif message["role"] == "assistant":
                    full_prompt.append(AIMessage(content=message["content"]))
        
        full_prompt.append(HumanMessage(content=query))
        gpt_answer = llm.invoke(full_prompt)
        return gpt_answer.content

# Example usage
if __name__ == "__main__":
    # Please set your own OpenAI API key here (api_key = "")
    content_processor = ChatbotOpenAI()
    
    # Test the chatbot with OpenAI
    llm_response = content_processor.answer(
        context="George Soros is a Hungarian-American investor, hedge fund manager, and philanthropist known for founding the Quantum Fund and popularizing the Theory of Reflexivity in finance.",
        query="Who is George Soros?",
        chat_history=[]
    )