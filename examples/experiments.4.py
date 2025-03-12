import zeroeval as ze
import time
from zeroeval.observability.decorators import span
from zeroeval.observability.tracer import tracer
import random
import openai
tracer.configure(
    flush_interval=5.0,   
    max_spans=50         
)

ze.init(api_key="zeval_CqvmJUkElthudZZF2kuoEBQ14TFV8SEw2hQtL76sehc")

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
      messages=[{"role": "user", "content": "Say a random animal name"}]
    )
    time.sleep(random.uniform(0.5, 1.5))
    return response.choices[0].message.content

@span(name="step_risky")
def step_risky(row):
    """
    This is a step that returns the input with random sleep time.
    """
    normal_step(row)
    time.sleep(random.uniform(0.5, 1.5))
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