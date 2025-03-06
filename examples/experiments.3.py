import zeroeval as ze
import time
from zeroeval.observability.decorators import span
from zeroeval.observability.tracer import tracer

tracer.configure(
    flush_interval=5.0,   
    max_spans=50         
)

ze.init(api_key="zeval_CqvmJUkElthudZZF2kuoEBQ14TFV8SEw2hQtL76sehc")

dataset = ze.Dataset.pull("Capitals")

@span(name="task")
def task(row):
    """
    This is a task that returns the input.
    """
    step1(row)
    step2(row)
    step3(row)
    return {"something": "hey", "other": 4}

@span(name="step1")
def step1(row):
    """
    This is a step that returns the input.
    """
    time.sleep(1)
    return {"something": "hey", "other": 4}

@span(name="step2")
def step2(row):
    """
    This is a step that returns the input.
    """
    time.sleep(2)
    return {"something": "hey", "other": 4}

@span(name="step3")
def step3(row):
    """
    This is a step that returns the input.
    """
    time.sleep(.5)
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