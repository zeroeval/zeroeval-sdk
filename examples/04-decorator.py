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

@ze.experiment(dataset=dataset)
def fn(x):
    print(x)
    return x
