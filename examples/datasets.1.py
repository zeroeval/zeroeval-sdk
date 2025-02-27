import zeroeval as ze

ze.init(api_key="zeval_CqvmJUkElthudZZF2kuoEBQ14TFV8SEw2hQtL76sehc")

dataset = ze.Dataset(
    name="Capitals-complex",
    description="A dataset for capital city classification",
    data=[
        {"input": {
          "country": "Colombia",
          "country_code": "CO"
        }, "output": {
          "capital": "Bogotá"
        }},
        {"input": {
          "country": "Peru",
          "country_code": "PE"
        }, "output": {
          "capital": "Lima"
        }},
        {
          "numero aqui": 123,
          "numero alla": 456
        },
        {
          "un array": [1, 2, 3],
          "un diccionario": {"a": 1, "b": 2},
          "una string": "hola",
          "un booleano": True,
          "un null": None
        }
    ]
)

dataset.push(create_new_version=True)