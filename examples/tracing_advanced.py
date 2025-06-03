from zeroeval.observability.decorators import span
from zeroeval.observability.tracer import tracer
import time
import uuid
import random

# Configure tracer  
tracer.configure(flush_interval=3.0, max_spans=100)

# Advanced Tracing Examples
# This showcases more complex tracing patterns

class MLPipeline:
    """Example ML Pipeline with comprehensive tracing"""
    
    def __init__(self, session_id=None):
        self.session_id = session_id or f"ml-pipeline-{uuid.uuid4().hex[:8]}"
        
    @span(name="load_model")
    def load_model(self, model_name):
        """Load ML model with tracing"""
        # Use the instance session_id by passing it to the decorator dynamically
        current_span = tracer.start_span(
            "load_model", 
            session_id=self.session_id,
            attributes={"model_name": model_name}
        )
        
        try:
            print(f"Loading model: {model_name}")
            time.sleep(0.3)  # Simulate model loading
            
            if random.random() < 0.1:  # 10% chance of failure
                raise Exception(f"Failed to load model {model_name}")
                
            current_span.set_io(
                input_data=f"model_name: {model_name}",
                output_data="model loaded successfully"
            )
            return {"model": model_name, "status": "loaded"}
            
        except Exception as e:
            current_span.set_error("ModelLoadError", str(e))
            raise
        finally:
            tracer.end_span(current_span)
    
    @span(name="preprocess_data")  
    def preprocess_data(self, data):
        """Preprocess data with nested span tracing"""
        current_span = tracer.start_span(
            "preprocess_data",
            session_id=self.session_id,
            attributes={"data_size": len(data)}
        )
        
        try:
            # Nested span for data cleaning
            cleaning_span = tracer.start_span(
                "clean_data",
                session_id=self.session_id,
                attributes={"operation": "remove_nulls"}
            )
            
            print("Cleaning data...")
            time.sleep(0.1)
            cleaned_data = [x for x in data if x is not None]
            tracer.end_span(cleaning_span)
            
            # Nested span for normalization
            norm_span = tracer.start_span(
                "normalize_data", 
                session_id=self.session_id,
                attributes={"operation": "scale_features"}
            )
            
            print("Normalizing data...")
            time.sleep(0.1) 
            normalized_data = [x / 100.0 for x in cleaned_data]
            tracer.end_span(norm_span)
            
            current_span.set_io(
                input_data=f"raw_data: {len(data)} items",
                output_data=f"processed_data: {len(normalized_data)} items"
            )
            
            return normalized_data
            
        except Exception as e:
            current_span.set_error("PreprocessingError", str(e))
            raise
        finally:
            tracer.end_span(current_span)
    
    @span(name="predict")
    def predict(self, data):
        """Make predictions with tracing"""
        current_span = tracer.start_span(
            "predict",
            session_id=self.session_id,
            attributes={
                "batch_size": len(data),
                "model_type": "neural_network"
            }
        )
        
        try:
            print(f"Making predictions for {len(data)} samples...")
            time.sleep(0.2)
            
            predictions = [random.random() for _ in data]
            
            current_span.set_io(
                input_data=f"features: {len(data)} samples", 
                output_data=f"predictions: {len(predictions)} results"
            )
            
            return predictions
            
        except Exception as e:
            current_span.set_error("PredictionError", str(e))
            raise
        finally:
            tracer.end_span(current_span)

# Context manager example for manual span control
class TimedOperation:
    """Context manager for timing operations with tracing"""
    
    def __init__(self, operation_name, session_id, attributes=None):
        self.operation_name = operation_name
        self.session_id = session_id
        self.attributes = attributes or {}
        self.span = None
        
    def __enter__(self):
        self.span = tracer.start_span(
            self.operation_name,
            session_id=self.session_id,
            attributes=self.attributes
        )
        return self.span
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.span.set_error(exc_type.__name__, str(exc_val))
        tracer.end_span(self.span)

@span(session_id="batch-processing-session", name="process_batch_advanced")
def process_batch_with_pipeline(batch_id, data):
    """Process a batch using the ML pipeline with comprehensive tracing"""
    
    session_id = f"batch-{batch_id}-session"
    pipeline = MLPipeline(session_id)
    
    print(f"\n🔄 Processing Batch {batch_id}")
    print(f"Session ID: {session_id}")
    
    try:
        # Step 1: Load model
        model_info = pipeline.load_model("bert-large-uncased")
        
        # Step 2: Preprocess data
        processed_data = pipeline.preprocess_data(data)
        
        # Step 3: Make predictions with manual timing
        with TimedOperation("prediction_timing", session_id, {"batch_id": batch_id}):
            predictions = pipeline.predict(processed_data)
            time.sleep(0.1)  # Additional processing time
        
        # Step 4: Post-process results
        with TimedOperation("postprocess_results", session_id, {"result_count": len(predictions)}):
            results = {
                "batch_id": batch_id,
                "model": model_info["model"],
                "predictions": len(predictions),
                "status": "success"
            }
            time.sleep(0.05)
        
        return results
        
    except Exception as e:
        print(f"❌ Batch {batch_id} failed: {e}")
        return {"batch_id": batch_id, "status": "failed", "error": str(e)}

def run_advanced_demo():
    """Run the advanced tracing demonstration"""
    
    print("🚀 Advanced Tracing Demo")
    print("This demonstrates:")
    print("  • Class-based tracing")
    print("  • Nested spans")
    print("  • Context managers") 
    print("  • Error handling")
    print("  • Manual span control")
    print("="*50)
    
    # Generate sample data
    sample_batches = [
        [10, 20, 30, None, 40, 50],  # Batch 1 (has null value)
        [15, 25, 35, 45, 55],        # Batch 2 (clean data)
        [5, 15, 25, 35],             # Batch 3 (smaller batch)
    ]
    
    results = []
    
    for i, batch_data in enumerate(sample_batches, 1):
        result = process_batch_with_pipeline(i, batch_data)
        results.append(result)
        print(f"✅ Batch {i} result: {result}")
        time.sleep(0.5)  # Delay between batches
    
    print("\n" + "="*50)
    print("📊 Summary:")
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    print(f"  • Successful batches: {successful}")
    print(f"  • Failed batches: {failed}")
    
    return results

if __name__ == "__main__":
    try:
        results = run_advanced_demo()
        
        print("\n📤 Flushing all tracing data...")
        tracer.flush()
        
        print("\n🎯 What to expect in the monitoring dashboard:")
        print("  • Multiple sessions (one per batch + main session)")
        print("  • Nested span hierarchies showing:")
        print("    - Model loading → Data preprocessing → Prediction → Post-processing")
        print("    - Sub-spans for data cleaning and normalization")
        print("  • Timing information for each operation")
        print("  • Error spans if model loading fails")
        print("  • Input/output data tracking")
        print("  • Custom attributes on spans")
        
        print("\n💡 Tips:")
        print("  • Run multiple times to see session accumulation")
        print("  • Check error rates in the dashboard") 
        print("  • Observe performance patterns across batches")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted - flushing data...")
        tracer.flush()
        print("✅ Data flushed successfully") 