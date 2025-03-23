import zeroeval as ze
import time
from zeroeval.observability.decorators import span
from zeroeval.observability.tracer import tracer
import random
import openai

tracer.configure(
    flush_interval=1.0,   
    max_spans=50         
)

ze.init(api_key="zeval_kRpse4itiDRECw6XLExpN3wbEI1LrylTlP23KzZjA1A")
client = openai.OpenAI(api_key="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")

dataset = ze.Dataset.pull("Capitals")

@span(name="calculate_capital")
def calculate_capital(row):
    """
    This is a step that returns the input with random sleep time.
    """
    response = client.chat.completions.create(
    model="gpt-4o",
      messages=[{"role": "user", "content": "What is the capital of " + row["input"] + "? Return only the name of the city in native language spelling, without any other text or punctuation."}]
    )
    return response.choices[0].message.content


def eval1(row, output):
    """
    Compare the output with the expected output
    """
    return row["expected_output"] == output

experiment = ze.Experiment(
    dataset=dataset,
    task=calculate_capital,
    evaluators=[eval1],
    parameters={
        "model": "gpt-4o",
        "prompt": "What is the capital of {{input}}? Return only the name of the city in native language spelling, without any other text or punctuation.",
        "embeddings_model": "text-embedding-3-small",
        "chunk_size": 100,
        "chunk_overlap": 20,
        "retrieval_top_k": 10,
        "reranker-model": "reranker-v2-small",
    }
)

experiment.run()