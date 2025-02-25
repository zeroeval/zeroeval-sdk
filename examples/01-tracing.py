from zeroeval.observability.decorators import span
from zeroeval.observability.tracer import tracer

import time

tracer.configure(
    flush_interval=5.0,   
    max_spans=50         
)

@span(name="process_user_data")
def process_user_data(user_id, data):
    time.sleep(1)
    return 2

@span(name="main")
def main():
    print("Starting main")
    process_user_data("user123", {"key": "value"})
    print("Finished main")
    
    with span(name="sub_span", attributes={"custom_attribute": "value"}):
        print("Starting sub_span")
        time.sleep(1)
        print("Finished sub_span")

main()
tracer.flush()