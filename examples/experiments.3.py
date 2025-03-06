import zeroeval as ze
import time
from zeroeval.observability.decorators import span
from zeroeval.observability.tracer import tracer
import random

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
    for step, func in [
        ("step1", step1),
        ("step2", step2),
        ("step3", step3)
    ]:
        choice = random.choices([0, 1, 2], weights=[0.2, 0.6, 0.2])[0]
        
        if choice >= 1:
            func(row)
        if choice == 2:
            func(row)  # Second execution

    return {"something": "hey", "other": 4}

@span(name="step1")
def step1(row):
    """
    This is a step that returns the input with random sleep time.
    """
    time.sleep(random.uniform(0.5, 1.5))
    return {"something": "hey", "other": 4}

@span(name="step2")
def step2(row):
    """
    This is a step that returns the input with random sleep time.
    """
    time.sleep(random.uniform(1.5, 2.5))
    return {"something": "hey", "other": 4}

@span(name="step3")
def step3(row):
    """
    This is a step that returns the input with random sleep time.
    """
    time.sleep(random.uniform(0.3, 0.7))
    return {"something": "hey", "other": 4}

def eval1(row, output):
    """
    Eval number one
    """
    print(row)
    return True

def eval2(row, output):
    """
    Eval number two
    """
    print(row)
    return "hey"

def eval3(row, output):
    """
    Eval number three
    """
    print(row)
    return [4,2,4,1]

experiment = ze.Experiment(
    dataset=dataset,
    task=task,
    evaluators=[eval1, eval2, eval3]
)

experiment.run()