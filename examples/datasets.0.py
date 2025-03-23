import zeroeval as ze

ze.init(api_key="zeval_CqvmJUkElthudZZF2kuoEBQ14TFV8SEw2hQtL76sehc")

dataset = ze.Dataset(
    name="Capitals",
    description="A dataset for capital city classification with capitals in their native language spelling",
    data=[
        {"input": "Colombia", "expected_output": "Bogotá"},  
        {"input": "Peru", "expected_output": "Lima"},  
        {"input": "Argentina", "expected_output": "Buenos Aires"},  
        {"input": "Chile", "expected_output": "Santiago"},  
        {"input": "Ecuador", "expected_output": "Quito"},  
        {"input": "Venezuela", "expected_output": "Caracas"},  
        {"input": "Brazil", "expected_output": "Brasília"},  
        {"input": "France", "expected_output": "Paris"},  
        {"input": "Germany", "expected_output": "Berlin"},  
        {"input": "Italy", "expected_output": "Roma"},  
        {"input": "Spain", "expected_output": "Madrid"},  
        {"input": "United Kingdom", "expected_output": "London"},  
        {"input": "Japan", "expected_output": "東京"},  
        {"input": "China", "expected_output": "北京"},  
        {"input": "India", "expected_output": "नई दिल्ली"},  
        {"input": "Australia", "expected_output": "Canberra"},  
        {"input": "Russia", "expected_output": "Москва"},  
        {"input": "Canada", "expected_output": "Ottawa"},  
        {"input": "Mexico", "expected_output": "Ciudad de México"},  
        {"input": "South Africa", "expected_output": "Pretoria"},  
        {"input": "Egypt", "expected_output": "القاهرة"},  
        {"input": "Kenya", "expected_output": "Nairobi"},
        {"input": "Philippines", "expected_output": "Maynila"},  
        {"input": "Nigeria", "expected_output": "Abuja"},   
        {"input": "Saudi Arabia", "expected_output": "الرياض"}, 
        {"input": "South Korea", "expected_output": "서울"},  
        {"input": "Thailand", "expected_output": "กรุงเทพมหานคร"}  
    ]
)

dataset.push()