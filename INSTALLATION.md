# Installation Guide

## Installation

```bash
# pip
pip install zeroeval

# uv
uv add zeroeval
```

That's it! The SDK automatically detects and instruments any supported frameworks you have installed.

## How Integrations Work

ZeroEval uses runtime detection to automatically instrument supported libraries **that you already have installed**. No additional dependencies needed!

When you import and initialize ZeroEval:

```python
import zeroeval as ze
ze.init()
```

It automatically detects and patches:

- **OpenAI** - if you have `openai` installed
- **LangChain** - if you have `langchain` installed
- **LangGraph** - if you have `langgraph` installed

## Example

```python
# Your existing code remains unchanged
import openai  # You installed this for your project
import zeroeval as ze

ze.init()  # This automatically patches OpenAI

client = openai.OpenAI()
# All OpenAI calls are now automatically traced!
```

## Running Examples

To run the examples in this repository, you'll need to install the frameworks they use:

```bash
# For OpenAI examples
pip install openai
# or with uv
uv add openai

# For LangChain examples
pip install langchain langchain-openai
# or with uv
uv add langchain langchain-openai

# For LangGraph examples
pip install langgraph
# or with uv
uv add langgraph
```

## Checking Active Integrations

To see which integrations are active:

```python
import zeroeval as ze
ze.init(debug=True)  # Logs which integrations were detected and activated
```

Output example:

```
[INFO] [zeroeval] Checking for available integrations: ['OpenAIIntegration', 'LangChainIntegration', 'LangGraphIntegration']
[INFO] [zeroeval] Setting up integration: OpenAIIntegration
[INFO] [zeroeval] Active integrations: ['OpenAIIntegration']
```
