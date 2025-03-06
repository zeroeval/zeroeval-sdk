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
        {"input": "Brazil", "output": "Brasília"},
        {"input": "France", "output": "Paris"},
        {"input": "Germany", "output": "Berlin"},
        {"input": "Italy", "output": "Rome"},
        {"input": "Spain", "output": "Madrid"},
        {"input": "United Kingdom", "output": "London"},
        {"input": "Japan", "output": "Tokyo"},
        {"input": "China", "output": "Beijing"},
        {"input": "India", "output": "New Delhi"},
        {"input": "Australia", "output": "Canberra"},
        {"input": "Russia", "output": "Moscow"},
        {"input": "Canada", "output": "Ottawa"},
        {"input": "Mexico", "output": "Mexico City"},
        {"input": "South Africa", "output": "Pretoria"},
        {"input": "Egypt", "output": "Cairo"},
        {"input": "Kenya", "output": "Nairobi"},
        {"input": "Nigeria", "output": "Abuja"},
        {"input": "Saudi Arabia", "output": "Riyadh"},
        {"input": "South Korea", "output": "Seoul"},
        {"input": "Thailand", "output": "Bangkok"}
    ]
)

# dataset.push()

dataset.push(create_new_version=True)
dataset.push(create_new_version=True)
dataset.push(create_new_version=True)

