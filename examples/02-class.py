import zeroeval as ze

ze.init(api_key="zeval_CqvmJUkElthudZZF2kuoEBQ14TFV8SEw2hQtL76sehc")

dataset = ze.Dataset(
    name="Capitals-6",
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

# def fn(x):
#     print(x)
#     return x
    
# experiment = ze.Experiment(dataset=dataset, task=fn, evaluators=[])

# experiment.run()