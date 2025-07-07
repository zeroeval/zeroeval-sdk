import zeroeval as ze

ze.init(api_key="sk_ze_sDaLKEbmov2O0eFML2ZNwIt40yvBJEIgFHyHXMmquPY")

dataset = ze.Dataset.pull("Capitals")


def task(row):
    """
    This is a task that returns the input.
    """
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
    return [4, 2, 4, 1]


experiment = ze.Experiment(dataset=dataset, task=task, evaluators=[eval1, eval2, eval3])

experiment.run()
