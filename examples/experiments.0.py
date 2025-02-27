import zeroeval as ze

ze.init(api_key="zeval_CqvmJUkElthudZZF2kuoEBQ14TFV8SEw2hQtL76sehc")

dataset = ze.Dataset.pull("Capitals")

def task(x):
    return x["input"]

experiment = ze.Experiment(
    dataset=dataset,
    task=task,
    evaluators=[]
)

results = experiment.run()
