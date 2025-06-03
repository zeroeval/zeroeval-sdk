import zeroeval as ze
from zeroeval.observability.decorators import span
from zeroeval.observability.tracer import tracer
import time
import random

# Initialize ZeroEval with API key
ze.init(api_key="sk_ze_f7mb9PQNbQEfOVSurY4S29B9YiUwrvO96Vi6QeicThU")

# Configure the tracer for more frequent flushing to see data quickly
tracer.configure(
    flush_interval=5.0,   
    max_spans=50         
)

# Example: Data Processing Pipeline with Session-based Tracing
# This showcases the Session → Trace → Span hierarchy

@span(session_id="data-pipeline-session-001", name="validate_data")
def validate_data(data):
    """Validate incoming data - this creates the first trace in our session"""
    time.sleep(0.1)
    
    if not data or len(data) == 0:
        raise ValueError("Data cannot be empty")
    
    return {"status": "valid", "record_count": len(data)}

@span(session_id="data-pipeline-session-001", name="transform_data")  
def transform_data(data):
    """Transform data - this creates spans in the same session but different traces"""
    time.sleep(0.2)
    
    transformed = []
    for item in data:
        # Each iteration could be a child span
        transformed.append({"id": item["id"], "value": item["value"] * 2})
    
    return transformed

@span(session_id="data-pipeline-session-001", name="save_data")
def save_data(data):
    """Save data - this demonstrates error scenarios"""
    time.sleep(0.1)
    
    # Simulate occasional failures
    if random.random() < 0.3:
        raise Exception("Database connection failed")
    
    return {"saved_records": len(data), "status": "success"}

@span(session_id="data-pipeline-session-001", name="process_batch")
def process_batch(batch_id, data):
    """Main processing function that orchestrates the pipeline"""
    print(f"Processing batch {batch_id} with {len(data)} records")
    
    try:
        # Step 1: Validate
        validation_result = validate_data(data)
        print(f"Validation: {validation_result}")
        
        # Step 2: Transform  
        transformed_data = transform_data(data)
        print(f"Transformed {len(transformed_data)} records")
        
        # Step 3: Save
        save_result = save_data(transformed_data)
        print(f"Save result: {save_result}")
        
        return {"batch_id": batch_id, "status": "completed", "records_processed": len(data)}
        
    except Exception as e:
        print(f"Batch {batch_id} failed: {e}")
        return {"batch_id": batch_id, "status": "failed", "error": str(e)}

# Example: Multiple sessions to show different workflows
@span(session_id="user-interaction-session-001", name="handle_user_request") 
def handle_user_request(user_id, request_type):
    """Handle user interaction - different session from data pipeline"""
    time.sleep(0.05)
    
    if request_type == "premium" and user_id % 2 == 0:
        # Simulate premium processing
        time.sleep(0.1)
        return {"user_id": user_id, "processed": True, "tier": "premium"}
    else:
        return {"user_id": user_id, "processed": True, "tier": "standard"}

@span(session_id="user-interaction-session-001", name="log_user_activity")
def log_user_activity(user_id, activity):
    """Log user activity - same session as user request"""
    time.sleep(0.02)
    return {"user_id": user_id, "activity_logged": True}

# Example: Manual span control for complex scenarios  
def manual_span_example():
    """Shows how to manually create spans for fine-grained control"""
    session_id = "manual-control-session-001"
    
    # Start a manual span
    span_obj = tracer.start_span(
        "complex_operation", 
        session_id=session_id,
        attributes={"operation_type": "manual", "complexity": "high"}
    )
    
    try:
        print("Starting complex manual operation...")
        time.sleep(0.3)
        
        # Set input/output data
        span_obj.set_io(
            input_data="complex input parameters",
            output_data="processed results"
        )
        
        # Simulate some work
        result = {"processed": True, "result_count": 42}
        print(f"Manual operation completed: {result}")
        return result
        
    except Exception as e:
        span_obj.set_error("ManualOperationError", str(e))
        raise
    finally:
        tracer.end_span(span_obj)

def main():
    print("🚀 Starting Enhanced Tracing Demo")
    print("This will create spans across multiple sessions that you can view in the live monitoring dashboard\n")
    
    # Demo 1: Data Pipeline Session
    print("📊 Demo 1: Data Processing Pipeline")
    print("Session ID: data-pipeline-session-001")
    
    sample_data = [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20}, 
        {"id": 3, "value": 30}
    ]
    
    # Process multiple batches in the same session
    for batch_num in range(1, 4):
        print(f"\n--- Processing Batch {batch_num} ---")
        result = process_batch(f"batch-{batch_num}", sample_data)
        print(f"Batch result: {result}")
        time.sleep(0.5)  # Small delay between batches
    
    print("\n" + "="*50)
    
    # Demo 2: User Interaction Session  
    print("👤 Demo 2: User Interaction Workflow")
    print("Session ID: user-interaction-session-001")
    
    users = [101, 102, 103, 104, 105]
    
    for user_id in users:
        print(f"\n--- Processing User {user_id} ---")
        request_type = "premium" if user_id % 3 == 0 else "standard"
        
        # Handle user request
        user_result = handle_user_request(user_id, request_type)
        print(f"User request result: {user_result}")
        
        # Log the activity
        log_result = log_user_activity(user_id, f"{request_type}_request")
        print(f"Activity logged: {log_result}")
        
        time.sleep(0.2)
    
    print("\n" + "="*50)
    
    # Demo 3: Manual Span Control
    print("🔧 Demo 3: Manual Span Control")
    print("Session ID: manual-control-session-001")
    
    manual_result = manual_span_example()
    print(f"Manual operation result: {manual_result}")
    
    print("\n" + "="*50)
    print("✅ Demo completed!")
    print("\n🔍 Check your live monitoring dashboard to see:")
    print("  • 3 Sessions created")
    print("    - data-pipeline-session-001 (batch processing)")
    print("    - user-interaction-session-001 (user requests)")  
    print("    - manual-control-session-001 (manual span)")
    print("  • Multiple traces within each session") 
    print("  • Spans showing the execution hierarchy")
    print("  • Error spans (if any database failures occurred)")
    print("  • Timing and performance metrics")
    print("  • Input/output data tracking")

if __name__ == "__main__":
    main()
    
    # Ensure all spans are flushed to backend
    print("\n📤 Flushing spans to backend...")
    tracer.flush()
    print("✅ All tracing data sent!")
    
    print("\n💡 Tip: Run this multiple times to generate more session data")
    print("    Each run will create new sessions with fresh data")