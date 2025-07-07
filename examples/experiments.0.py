import zeroeval as ze

ze.init(api_key="sk_ze_sDaLKEbmov2O0eFML2ZNwIt40yvBJEIgFHyHXMmquPY")

dataset = ze.Dataset.pull("Capitals")

def task(x):
    return x["input"]

experiment = ze.Experiment(
    dataset=dataset,
    task=task,
    evaluators=[]
)

results = experiment.run()
