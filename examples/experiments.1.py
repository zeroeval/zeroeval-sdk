import zeroeval as ze

ze.init(api_key="sk_ze_4OxO2q-uR6beq32qxV-zPkq0uONq4CIjtS_Bc7P9idM")

dataset = ze.Dataset.pull("Capitals")


def task(row):
    """
    This is a task that returns the input.
    """
    return row["input"]

def exact_match(row, output):
    """
    This is an evaluator that returns the exact match between the input and the output.
    """
    print(row)
    return row["input"] == output

experiment = ze.Experiment(
    dataset=dataset,
    task=task,
    evaluators=[exact_match]
)


# test_results = experiment.run(dataset[:1])
# all_results = experiment.run()

results = experiment.run_task()
results = experiment.run_evaluators([exact_match])

