# Task Auto-Creation with ze.prompt()

## Summary

**Tasks ARE automatically created** when using `ze.prompt()`. The previous warning messages in the SDK were incorrect and have been fixed.

## How It Works

1. **SDK Side** (`ze.prompt()`):
   - Creates a prompt version in the backend via `/v1/tasks/{task_name}/prompt/versions/ensure`
   - This endpoint creates a **prompt** (not a task yet) if it doesn't exist
   - Wraps the prompt content with `<zeroeval>` metadata containing the task name
   - Returns the wrapped prompt to be used in OpenAI calls

2. **OpenAI Integration**:
   - Intercepts the OpenAI API call
   - Extracts the `<zeroeval>` metadata
   - Adds task information to the span attributes
   - Sends the span to the ZeroEval backend

3. **Backend Enrichment** (Asynchronous):
   - The span is received and queued for processing
   - Celery worker picks up the `process_span` task
   - Calls `_process_task_association()` which:
     - Looks for task metadata in span attributes
     - Checks if task exists in database
     - **Automatically creates the task** if it doesn't exist
     - Links the span to the task

## Why You Might Not See the Task Immediately

The task creation happens **asynchronously** via background workers:

1. Your code calls OpenAI with the `ze.prompt()` wrapped prompt
2. The span is sent to the backend
3. A Celery worker processes the span (this can take a few seconds)
4. The task is created during span enrichment

**Expected delay**: 1-5 seconds typically, depending on backend load.

## Fixed Warning Messages

Previously, the SDK logged:
```
WARNING: Task ID 'bookstore-support-agent' found in zeroeval_prompt. 
Note: zeroeval_prompt does NOT automatically create or update tasks.
```

This was **incorrect**. The updated message now says:
```
INFO: Task ID 'bookstore-support-agent' found in zeroeval_prompt. 
The task will be automatically created if it doesn't exist yet.
```

## Verification

To verify task creation:

1. Run your code with `ze.prompt()`
2. Wait 5-10 seconds for async processing
3. Check the ZeroEval dashboard - the task should appear under "Tuning" or "Tasks"
4. Check backend logs for: `[TASK CREATION] Created task 'task-name'`

## Code Location

- **SDK**: `zeroeval-sdk/src/zeroeval/__init__.py` - `prompt()` function
- **Backend**: `backend/src/tasks/object_enrichment.py` - `_process_task_association()` and `_create_task_from_slug()`
- **API Endpoint**: `backend/src/routes/prompts_route.py` - `/v1/tasks/{task_name}/prompt/versions/ensure`

## Changes Made

1. Updated `zeroeval_prompt()` docstring to reflect auto-creation
2. Changed WARNING to INFO log level
3. Updated log messages to indicate tasks are created automatically
4. Files modified:
   - `zeroeval-sdk/src/zeroeval/observability/integrations/openai/integration.py`

