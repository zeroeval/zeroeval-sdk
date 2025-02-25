import zeroeval as ze

dataset = ze.Dataset(
    name="Capitals",
    description="A dataset for capital city classification",
    data=[
        {"input": "Colombia", "output": "Bogotá"},
        {"input": "Peru", "output": "Lima"},
        {"input": "Argentina", "output": "Buenos Aires"},
        {"input": "Chile", "output": "Santiago"},
        {"input": "Ecuador", "output": "Quito"},
        {"input": "Venezuela", "output": "Caracas"},
    ]
)

dataset.push()

def fn(x):
    print(x)
    return x
    
experiment = ze.Experiment(dataset=dataset, task=fn, evaluators=[])

experiment.run()