import zeroeval as ze

@ze.exp(dataset="MNIST", model="ClassificationModel")
def train():
    print("Training the model...")

@ze.exp(dataset="CIFAR-10", model="AnotherModel")
def evaluate():
    print("Evaluating on CIFAR-10...")

print("running prod")