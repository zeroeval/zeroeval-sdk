# ZeroEval Tracing Examples

These examples demonstrate the enhanced session-based tracing system that creates a proper hierarchy of **Session → Trace → Span** for better observability and monitoring.

## 🎯 What You'll Learn

- How to use the `@span` decorator with session IDs
- Creating organized traces within sessions
- Handling errors and performance monitoring
- Viewing your data in the live monitoring dashboard

## 📁 Example Files

### 1. `tracing.0.py` - Basic Session-Based Tracing

**Perfect for getting started**

Shows:

- ✅ Simple `@span` decorator usage with `session_id`
- ✅ Multiple workflows in different sessions
- ✅ Error simulation and handling
- ✅ Real-world data processing pipeline

**What it creates:**

- **2 Sessions**:
  - `data-pipeline-session-001` (data processing workflow)
  - `user-interaction-session-001` (user request handling)
- **Multiple Traces** per session (one per function call)
- **Performance metrics** and error tracking

```python
@span(session_id="data-pipeline-session-001", name="validate_data")
def validate_data(data):
    # Your function logic here
    pass
```

### 2. `tracing_advanced.py` - Advanced Patterns

**For complex scenarios**

Shows:

- ✅ Class-based tracing with instance session IDs
- ✅ Nested spans and hierarchical tracing
- ✅ Context managers for manual span control
- ✅ Input/output data tracking
- ✅ Custom attributes and metadata

**What it creates:**

- **Dynamic Sessions** (one per batch)
- **Nested Span Hierarchies**:
  ```
  process_batch_advanced
  ├── load_model
  ├── preprocess_data
  │   ├── clean_data
  │   └── normalize_data
  ├── prediction_timing
  │   └── predict
  └── postprocess_results
  ```
- **Rich metadata** and performance insights

## 🚀 Getting Started

### 1. Set up your environment

```bash
# Make sure you have the ZeroEval SDK installed
cd sdk/
pip install -e .

# Set your backend URL (if different from default)
export API_URL="https://api.zeroeval.com"
export API_KEY="your-api-key"  # Optional, for authentication
```

### 2. Run the basic example

```bash
cd sdk/examples/
python tracing.0.py
```

**Expected output:**

```
🚀 Starting Enhanced Tracing Demo
📊 Demo 1: Data Processing Pipeline
Session ID: data-pipeline-session-001

--- Processing Batch 1 ---
Processing batch batch-1 with 3 records
Validation: {'status': 'valid', 'record_count': 3}
...
✅ All tracing data sent!
```

### 3. Run the advanced example

```bash
python tracing_advanced.py
```

### 4. View in Live Monitoring Dashboard

Navigate to your frontend monitoring dashboard:

```
http://localhost:3000/[your-team-id]/monitoring/live
```

## 📊 What You'll See in the Dashboard

### Sessions View

The live monitoring dashboard will show:

| Session ID                   | Name            | Attributes | Created             |
| ---------------------------- | --------------- | ---------- | ------------------- |
| data-pipeline-session-001    | Unnamed Session | {}         | 2024-01-15 10:30:25 |
| user-interaction-session-001 | Unnamed Session | {}         | 2024-01-15 10:30:28 |
| batch-1-session              | Unnamed Session | {}         | 2024-01-15 10:35:12 |

### Session Statistics

- **Total Sessions**: Number of unique sessions created
- **Total Traces**: All traces across sessions
- **Total Spans**: All spans across all traces
- **Total Errors**: Count of failed operations
- **Avg Session Duration**: Average time per session
- **P95 Trace Duration**: 95th percentile trace timing

### Interactive Features

- 🔍 **Click on sessions** to explore their traces
- 📅 **Filter by date range** to see specific time periods
- ⟳ **Auto-refresh** every 30 seconds for live data
- 📄 **Pagination** for large datasets

## 🛠️ Customizing Your Tracing

### Session Management

```python
# Explicit session ID (recommended for workflows)
@span(session_id="my-workflow-001", name="step1")
def my_function():
    pass

# Auto-generated session (for independent operations)
@span(name="standalone_operation")
def independent_function():
    pass  # Creates new session automatically
```

### Error Handling

```python
@span(session_id="error-demo", name="risky_operation")
def risky_function():
    try:
        # Your code here
        if some_condition:
            raise ValueError("Something went wrong")
    except Exception as e:
        # Errors are automatically captured by the span
        raise
```

### Adding Metadata

```python
@span(
    session_id="metadata-demo",
    name="enriched_operation",
    attributes={"version": "1.0", "env": "production"}
)
def enriched_function(user_id):
    pass
```

### Manual Span Control

```python
# For complex scenarios where decorator isn't enough
current_span = tracer.start_span(
    "manual_operation",
    session_id="manual-session",
    attributes={"custom": "value"}
)

try:
    # Your operation
    result = do_something()
    current_span.set_io(
        input_data="description of input",
        output_data="description of output"
    )
except Exception as e:
    current_span.set_error("ErrorType", str(e))
    raise
finally:
    tracer.end_span(current_span)
```

## 🔧 Configuration Options

### Tracer Configuration

```python
tracer.configure(
    flush_interval=5.0,  # Flush to backend every 5 seconds
    max_spans=100        # Flush when 100 spans are buffered
)
```

### Environment Variables

```bash
# Backend URL (default: https://api.zeroeval.com)
export API_URL="https://your-backend.com"

# API Key for authentication (optional)
export API_KEY="your-api-key"
```

## 🎯 Best Practices

### 1. Use Meaningful Session IDs

```python
# ✅ Good: Describes the workflow
session_id = f"user-{user_id}-onboarding"
session_id = f"batch-processing-{date}"
session_id = f"ml-training-experiment-{experiment_id}"

# ❌ Avoid: Generic or unclear
session_id = "session-1"
session_id = str(uuid.uuid4())
```

### 2. Group Related Operations

```python
# ✅ Good: All steps in same session
@span(session_id="data-pipeline-001", name="extract")
def extract_data(): pass

@span(session_id="data-pipeline-001", name="transform")
def transform_data(): pass

@span(session_id="data-pipeline-001", name="load")
def load_data(): pass
```

### 3. Handle Errors Gracefully

```python
@span(session_id="robust-session", name="safe_operation")
def safe_operation():
    try:
        risky_code()
    except SpecificError as e:
        # Log but don't re-raise if recoverable
        print(f"Handled error: {e}")
        return default_value()
    except Exception:
        # Re-raise unexpected errors
        raise
```

### 4. Add Context with Attributes

```python
@span(
    session_id="context-rich",
    name="process_user_data",
    attributes={
        "user_tier": "premium",
        "data_size": "large",
        "processing_mode": "batch"
    }
)
def process_user_data(user_data):
    pass
```

## 🐛 Troubleshooting

### Not seeing data in dashboard?

1. Check that `tracer.flush()` is called
2. Verify `API_URL` environment variable
3. Ensure backend is running and accessible
4. Check console for any error messages

### Sessions not grouping correctly?

1. Ensure consistent `session_id` across related operations
2. Check for typos in session ID strings
3. Verify spans are in the same thread/process

### Performance impact?

1. Adjust `flush_interval` for your needs
2. Monitor `max_spans` buffer size
3. Use sampling for high-volume operations

## 📚 Next Steps

1. **Explore the Dashboard**: Run the examples and navigate the live monitoring interface
2. **Integrate with Your Code**: Add `@span` decorators to your existing functions
3. **Set Up Alerts**: Use the monitoring data to create performance alerts
4. **Advanced Analysis**: Build custom dashboards for your specific metrics

---

Happy tracing! 🎉

For more advanced usage, check out the [ZeroEval Documentation](../README.md).
