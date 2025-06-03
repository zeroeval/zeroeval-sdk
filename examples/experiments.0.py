import zeroeval as ze

ze.init(api_key="sk_ze_rDMKmDkyHuc_OXykkWUuwqUtlGNx2auUgNifP5THobw")

dataset = ze.Dataset.pull("Capitals")

def task(x):
    return x["input"]

experiment = ze.Experiment(
    dataset=dataset,
    task=task,
    evaluators=[]
)

results = experiment.run()
