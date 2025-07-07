import json
import random
import time

import openai

import zeroeval as ze
from zeroeval.observability import span

# --- Configuration ---
# 1. Initialize ZeroEval.
#    It's recommended to set your API key as an environment variable:
#    export ZEROEVAL_API_KEY="sk_ze_..."
ze.init(
    debug=True,
    api_key="sk_ze_diTPEUddB7MHvWGSA_NeZEseRd_1_ID3LOwfch0TxQg",
    api_url="http://localhost:8000",
)  # Set debug=True to see detailed tracer logs.


# Create an OpenAI client
client = openai.OpenAI(
    api_key="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"
)


# --- Small utility operations with varying durations ---


@span(name="validate_input")
def validate_input(text: str):
    """Quick input validation - 2ms operation"""
    time.sleep(0.002)
    return bool(text and len(text.strip()) > 0)


@span(name="sanitize_text")
def sanitize_text(text: str):
    """Text sanitization - 10ms operation"""
    time.sleep(0.01)
    return text.strip().replace("\n", " ")


@span(name="check_rate_limits")
def check_rate_limits():
    """Rate limit checking - 2ms operation"""
    time.sleep(0.002)
    return {"remaining": 1000, "reset_time": time.time() + 3600}


@span(name="log_request_start")
def log_request_start(operation: str):
    """Request logging - 2ms operation"""
    time.sleep(0.002)
    return {"timestamp": time.time(), "operation": operation}


@span(name="prepare_prompt_metadata")
def prepare_prompt_metadata(content: str):
    """Metadata preparation - 10ms operation"""
    time.sleep(0.01)
    return {
        "length": len(content),
        "word_count": len(content.split()),
        "timestamp": time.time(),
    }


@span(name="cache_lookup")
def cache_lookup(key: str):
    """Cache lookup simulation - 50ms operation"""
    time.sleep(0.05)


@span(name="format_request")
def format_request(messages: list):
    """Request formatting - 10ms operation"""
    time.sleep(0.01)
    return json.dumps(messages, indent=2)


@span(name="parse_response")
def parse_response(response):
    """Response parsing - 10ms operation"""
    time.sleep(0.01)
    return response.choices[0].message.content


@span(name="validate_response")
def validate_response(content: str):
    """Response validation - 2ms operation"""
    time.sleep(0.002)
    return content is not None and len(content) > 0


@span(name="cache_store")
def cache_store(key: str, value: str):
    """Cache storage - 50ms operation"""
    time.sleep(0.05)
    return True


@span(name="log_request_end")
def log_request_end(operation: str, success: bool):
    """Request completion logging - 2ms operation"""
    time.sleep(0.002)
    return {"operation": operation, "success": success, "end_time": time.time()}


@span(name="update_metrics")
def update_metrics(operation: str, duration: float):
    """Metrics update - 10ms operation"""
    time.sleep(0.01)
    return {"operation": operation, "duration": duration, "recorded": True}


@span(name="text_preprocessing")
def text_preprocessing(text: str):
    """Text preprocessing - 50ms operation"""
    time.sleep(0.05)
    return text.lower().strip()


@span(name="extract_keywords")
def extract_keywords(text: str):
    """Keyword extraction - 10ms operation"""
    time.sleep(0.01)
    words = text.split()
    return [word for word in words if len(word) > 4][:5]


@span(name="content_filter")
def content_filter(text: str):
    """Content filtering - 2ms operation"""
    time.sleep(0.002)
    return not any(word in text.lower() for word in ["spam", "inappropriate"])


@span(name="performance_monitor")
def performance_monitor():
    """Performance monitoring - 2ms operation"""
    time.sleep(0.002)
    return {"cpu_usage": random.uniform(10, 30), "memory_usage": random.uniform(40, 60)}


# --- Application Logic with Tracing ---
# Each function is decorated with @span to capture its execution as a trace.


@span(name="generate_story_idea")
def generate_story_idea():
    """(1st LLM Call) Generates a random story idea using GPT-3.5."""
    print("1. Generating a story idea...")

    # Pre-processing mini operations
    validate_input("Give me a story idea")
    check_rate_limits()
    log_request_start("generate_story_idea")
    cache_lookup("story_idea_cache")
    performance_monitor()

    # Main LLM call
    messages = [
        {
            "role": "user",
            "content": "Give me a one-sentence story idea about AI in the future.",
        }
    ]
    prepare_prompt_metadata(messages[0]["content"])
    format_request(messages)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    # Post-processing mini operations
    idea = parse_response(response)
    validate_response(idea)
    content_filter(idea)
    cache_store("story_idea", idea)
    log_request_end("generate_story_idea", True)
    update_metrics("generate_story_idea", 0.5)

    print(f"   -> Idea: {idea}\n")
    return idea


@span(name="expand_on_idea")
def expand_on_idea(idea: str):
    """(2nd LLM Call) Expands the story idea into a short paragraph."""
    print("2. Expanding the idea into a paragraph...")

    # Pre-processing mini operations
    validate_input(idea)
    sanitize_text(idea)
    check_rate_limits()
    log_request_start("expand_on_idea")
    cache_lookup(f"expanded_{hash(idea)}")
    text_preprocessing(idea)

    # Main LLM call
    messages = [
        {
            "role": "user",
            "content": f"Expand this idea into a short, three-sentence paragraph: {idea}",
        }
    ]
    prepare_prompt_metadata(messages[0]["content"])
    format_request(messages)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    # Post-processing mini operations
    paragraph = parse_response(response)
    validate_response(paragraph)
    extract_keywords(paragraph)
    content_filter(paragraph)
    cache_store(f"expanded_{hash(idea)}", paragraph)
    log_request_end("expand_on_idea", True)
    update_metrics("expand_on_idea", 0.7)
    performance_monitor()

    print(f"   -> Paragraph: {paragraph}\n")
    return paragraph


@span(name="extract_main_character")
def extract_main_character(paragraph: str):
    """(Intermediate Processing) A simple, non-LLM step to process text."""
    print("3. Extracting main character (local processing)...")

    # Pre-processing mini operations
    validate_input(paragraph)
    sanitize_text(paragraph)
    text_preprocessing(paragraph)

    # In a real app, this could be complex logic. Here, we just find a proper noun.
    words = paragraph.split()
    character = next(
        (
            word.strip(".,")
            for word in words
            if word[0].isupper() and word.lower() != "a"
        ),
        "the AI",
    )

    # Post-processing mini operations
    validate_response(character)
    log_request_end("extract_main_character", True)

    print(f"   -> Character: {character}\n")
    return character


@span(name="write_final_title")
def write_final_title(paragraph: str, character: str):
    """(3rd LLM Call) Writes a final title using the paragraph and character."""
    print("4. Writing a final title...")

    # Pre-processing mini operations
    validate_input(paragraph)
    validate_input(character)
    check_rate_limits()
    log_request_start("write_final_title")
    cache_lookup(f"title_{hash(paragraph + character)}")
    performance_monitor()

    prompt = f"""
    Based on the following story and main character, write a creative title.

    Story: {paragraph}
    Main Character: {character}
    """

    prepare_prompt_metadata(prompt)
    messages = [{"role": "user", "content": prompt}]
    format_request(messages)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    # Post-processing mini operations
    title = parse_response(response).strip('"')
    validate_response(title)
    sanitize_text(title)
    content_filter(title)
    cache_store(f"title_{hash(paragraph + character)}", title)
    log_request_end("write_final_title", True)
    update_metrics("write_final_title", 0.6)

    print(f"   -> Title: {title}\n")
    return title


def main():
    """
    Runs a simple LLM chain that generates an idea, expands it, processes it,
    and creates a title. Each step is traced automatically with many micro-operations.
    """
    print("🚀 Starting Complex OpenAI Tracing Example with Micro-Operations...")
    print("Each step will be captured as a span in your ZeroEval workspace.\n")

    # The main application logic, wrapped in a single parent span.
    with span(name="full_story_generation"):
        # Initial setup operations
        performance_monitor()
        check_rate_limits()

        idea = generate_story_idea()
        paragraph = expand_on_idea(idea)
        character = extract_main_character(paragraph)
        title = write_final_title(paragraph, character)

        # Final cleanup operations
        update_metrics("full_story_generation", 2.0)
        performance_monitor()

    print("-" * 50)
    print("✅ Chain completed!")
    print(f"\nGenerated Story Title: {title}")
    print(
        "\n🔍 Check your ZeroEval dashboard to see the full, nested trace with micro-operations."
    )


if __name__ == "__main__":
    main()
