import zeroeval as ze

ze.init(api_key="sk_ze_4OxO2q-uR6beq32qxV-zPkq0uONq4CIjtS_Bc7P9idM")

dataset = ze.Dataset.pull("Capitals")

def task(x):
    return x["input"]

experiment = ze.Experiment(
    dataset=dataset,
    task=task,
    evaluators=[]
)

results = experiment.run()
