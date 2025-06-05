import zeroeval as ze

ze.init(api_key="sk_ze_4OxO2q-uR6beq32qxV-zPkq0uONq4CIjtS_Bc7P9idM")

dataset = ze.Dataset(
    name="Conversation",
    description="A dataset for conversation",
    data=[
        {
            "input": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that provides information about capital cities."},
                    {"role": "user", "content": "What is the capital of France?"}
                ]
            },
            "output": {
                "messages": [
                    {"role": "assistant", "content": "The capital of France is Paris."}
                ]
            },
        },
        {
            "input": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that provides information about capital cities."},
                    {"role": "user", "content": "What's the capital of Japan?"},
                    {"role": "assistant", "content": "The capital of Japan is Tokyo."},
                    {"role": "user", "content": "And what about South Korea?"}
                ]
            },
            "output": {
                "messages": [
                    {"role": "assistant", "content": "The capital of South Korea is Seoul."}
                ]
            }
        },
        {
            "input": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that provides information about capital cities."},
                    {"role": "user", "content": "Tell me about the capital of Brazil."},
                    {"role": "assistant", "content": "The capital of Brazil is Brasília. It was inaugurated in 1960, replacing Rio de Janeiro as the country's capital."},
                    {"role": "user", "content": "What's interesting about it?"}
                ]
            },
            "output": {
                "messages": [
                    {"role": "assistant", "content": "Brasília is known for its unique modernist architecture designed by Oscar Niemeyer and urban planning by Lúcio Costa. The city's airplane-shaped layout and distinctive buildings earned it UNESCO World Heritage status in 1987. It was a planned city built from scratch in just 41 months, which was an impressive feat of engineering and construction."}
                ]
            }
        }
    ]
)

dataset.push()