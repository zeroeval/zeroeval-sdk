import zeroeval as ze

ze.init(api_key="zeval_CqvmJUkElthudZZF2kuoEBQ14TFV8SEw2hQtL76sehc")

dataset = ze.Dataset(
    name="Capitals",
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

dataset.push(create_new_version=True)
dataset.push(create_new_version=True)
dataset.push(create_new_version=True)

# 3 versions should be created