from ragas import evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, AnswerCorrectness, ContextRecall, ContextPrecision
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import dotenv_values
from ragas.embeddings import LangchainEmbeddingsWrapper

data = {
    "question": ["Who wrote 1984?"],
    "answer": ["1984 was written by George Orwell."],
    "contexts": [["1984 is a dystopian novel written by George Orwell in 1949."]],
    "ground_truth": ["George Orwell wrote the novel 1984."]
}
dataset = Dataset.from_dict(data)
config = dotenv_values(".env")
llm = LangchainLLMWrapper(ChatOpenAI(api_key=config["OPENAI_API_KEY"], model="gpt-4o-mini", temperature=0.1))
embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=config["OPENAI_API_KEY"], model="text-embedding-3-small"))

metrics = [
    Faithfulness(),        # Groundedness
    ResponseRelevancy(),   # Efficacy
    AnswerCorrectness(),   # Coherence
    ContextRecall(),       # Completeness
    ContextPrecision()     # Precision
]

results = evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings, show_progress=False)
print(results)