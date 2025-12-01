# rag_eval_given_preds.py
import json
from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator
from dotenv import dotenv_values
config = dotenv_values(".env")

# -----------------------------
# 1. Init LLM Judge
# -----------------------------
llm = ChatOpenAI(api_key=config["OPEN_AI_API_KEY"], model="gpt-4o-mini", temperature=0)

# ------------------------------
# Define evaluators
# ------------------------------

# Reference-free
faithfulness_criteria = {
    "faithfulness": "Does the prediction rely only on the provided context, without hallucinations?"
}
faithfulness_eval = load_evaluator("criteria", criteria=faithfulness_criteria, llm=llm)
conciseness_eval = load_evaluator("criteria", criteria="conciseness", llm=llm)
relevance_eval = load_evaluator("criteria", criteria="relevance", llm=llm)
coherence_eval = load_evaluator("criteria", criteria="coherence", llm=llm)
harmfulness_eval = load_evaluator("criteria", criteria="harmfulness", llm=llm)

# Reference-based
correctness_eval = load_evaluator("labeled_criteria", criteria="correctness", llm=llm)

# ------------------------------
# Example dataset
# ------------------------------
dataset = [
    {
        "model":"4o",
        "temperature":"0.1",
        "input": "Who wrote the novel 1984?",
        "prediction": "George Orwell wrote '1984', a dystopian novel about surveillance.",
        "context": ["George Orwell wrote '1984', a dystopian novel about totalitarianism and surveillance."],
        "reference": "George Orwell"
    },
    {
        "model":"4o",
        "temperature":"0.1",
        "input": "When was the Eiffel Tower completed?",
        "prediction": "The Eiffel Tower was finished in 1889 for the Paris exposition. And then people came to see it" ,
        "context": ["The Eiffel Tower was completed in 1889 for the Exposition Universelle in Paris."],
        "reference": "1889"
    }
]

# ------------------------------
# Run evaluations
# ------------------------------
results = []
for ex in dataset:
    ctx_input = f"Question: {ex['input']}\n\nContext:\n" + "\n".join(ex["context"])

    evals = {
        "faithfulness": faithfulness_eval.evaluate_strings(input=ctx_input, prediction=ex["prediction"]),
        "conciseness": conciseness_eval.evaluate_strings(input=ex["input"], prediction=ex["prediction"]),
        "relevance": relevance_eval.evaluate_strings(input=ex["input"], prediction=ex["prediction"]),
        "coherence": coherence_eval.evaluate_strings(input=ex["input"], prediction=ex["prediction"]),
        "harmfulness": harmfulness_eval.evaluate_strings(input=ex["input"], prediction=ex["prediction"]),
        "correctness": correctness_eval.evaluate_strings(
            input=ex["input"], prediction=ex["prediction"], reference=ex["reference"]
        )
    }

    results.append({
        "model": ex["model"],
        "question": ex["input"],
        "prediction": ex["prediction"],
        **evals
    })

# ------------------------------
# Print results
# ------------------------------
print(json.dumps(results, indent=2))