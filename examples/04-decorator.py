import zeroeval as ze


@ze.experiment(dataset="Capitals-10")
def fn(x):
    print(x)
    return x


# Run with `zeroeval run 04-decorator.py`
