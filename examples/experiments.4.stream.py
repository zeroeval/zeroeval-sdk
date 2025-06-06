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

ze.init(api_key="sk_ze_sDaLKEbmov2O0eFML2ZNwIt40yvBJEIgFHyHXMmquPY")

dataset = ze.Dataset.pull("Capitals")

@span(name="task")
def task(row):
    """
    This is a task that randomly executes, skips, or doubles steps.
    """
    # For each step, randomly decide: 0 = skip, 1 = run once, 2 = run twice
    step_risky(row)
    return "success"

@span(name="normal_step")
def normal_step(row):
    """
    This is a step that returns the input with random sleep time.
    """
    client = openai.OpenAI(api_key="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")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Tell a 1 line story about a bear, start with 'Once upon a time'."}],
        stream=True  # Enable streaming
    )
    
    # Collect the streamed response
    full_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
            
    time.sleep(random.uniform(0.1, .2))
    return full_response

@span(name="step_risky")
def step_risky(row):
    """
    This is a step that returns the input with random sleep time.
    """
    normal_step(row)
    time.sleep(random.uniform(.2, .4))
    normal_step(row)
    if random.random() < 0.5:
        error_step(row)
    return "success"

@span(name="error_step")
def error_step(row):
    """
    This is a step that returns the input with random sleep time.
    """
    raise Exception("This is an error step")



def eval1(row, output):
    """
    Eval number one
    """
    print(row)
    return random.random()

experiment = ze.Experiment(
    dataset=dataset,
    task=task,
    evaluators=[eval1]
)

experiment.run()