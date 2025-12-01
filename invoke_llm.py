from langchain_openai import ChatOpenAI
from dotenv import dotenv_values

class InvokeLLM:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def invoke_llm(self, prompt: str) -> str:
        # Placeholder for LLM invocation logic
        config = dotenv_values(".env")
        llm = ChatOpenAI(model_name=self.model_name, 
                         openai_api_key=config["OPENAI_API_KEY"],
                         temperature=0.1)
        response = llm.invoke(prompt)
        return response.content
    
if __name__ == "__main__":
    invoke_llm = InvokeLLM(model_name="gpt-4o")
    result = invoke_llm.invoke_llm("What is the Capital of France?")
    print(result)