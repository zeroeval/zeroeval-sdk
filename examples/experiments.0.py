import zeroeval as ze

ze.init(api_key="sk_ze_YYP_c-VAd909JUr3cwrKXtYCYO8ppOgUP4CTGPeiij0")

dataset = ze.Dataset.pull("Capitals")

def task(x):
    return x["input"]

experiment = ze.Experiment(
    dataset=dataset,
    task=task,
    evaluators=[]
)

results = experiment.run()
