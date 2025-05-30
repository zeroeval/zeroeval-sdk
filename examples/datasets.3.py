import zeroeval as ze

ze.init(api_key="sk_ze_YYP_c-VAd909JUr3cwrKXtYCYO8ppOgUP4CTGPeiij0")

dataset = ze.Dataset(
    name="Capitals-again",
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

dataset = ze.Dataset.pull("Capitals-again")

dataset.add_rows(
    [
        {
            "input": "Brazil",
            "output": "Brasília"
        },
        {
            "input": "Argentina",
            "output": "Buenos Aires"
        }
    ]
)  

dataset.delete_row(0)

dataset.push(create_new_version=True)

print(dataset[0])

dataset.update_row(
    0,
    {
        "input": dataset[0]["input"],
        "output": "Mexico City"
    }
)

print(dataset[2:3])

dataset[4] = {
    "input": "Mexico",
    "output": "Mexico City"
}

del dataset[3]

dataset.push(create_new_version=True)

print(dataset)
print(dataset.data)

# Result should be:
"""
        {"input": "Peru", "output": "Mexico City"},
        {"input": "Argentina", "output": "Buenos Aires"},
        {"input": "Chile", "output": "Santiago"}, <- print
        {"input": "Mexico", "output": "Mexico City"},
        {"input": "Brazil", "output": "Brasília"},
        {"input": "Argentina", "output": "Buenos Aires"}
"""