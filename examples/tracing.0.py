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


# --- Application Logic with Tracing ---
# Each function is decorated with @span to capture its execution as a trace.


@span(name="generate_story_idea")
def generate_story_idea():
    """(1st LLM Call) Generates a random story idea using GPT-3.5."""
    print("1. Generating a story idea...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Give me a one-sentence story idea about AI in the future.",
            }
        ],
    )
    idea = response.choices[0].message.content
    print(f"   -> Idea: {idea}\n")
    return idea


@span(name="expand_on_idea")
def expand_on_idea(idea: str):
    """(2nd LLM Call) Expands the story idea into a short paragraph."""
    print("2. Expanding the idea into a paragraph...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Expand this idea into a short, three-sentence paragraph: {idea}",
            }
        ],
    )
    paragraph = response.choices[0].message.content
    print(f"   -> Paragraph: {paragraph}\n")
    return paragraph


@span(name="extract_main_character")
def extract_main_character(paragraph: str):
    """(Intermediate Processing) A simple, non-LLM step to process text."""
    print("3. Extracting main character (local processing)...")
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
    print(f"   -> Character: {character}\n")
    return character


@span(name="write_final_title")
def write_final_title(paragraph: str, character: str):
    """(3rd LLM Call) Writes a final title using the paragraph and character."""
    print("4. Writing a final title...")
    prompt = f"""
    Based on the following story and main character, write a creative title.

    Story: {paragraph}
    Main Character: {character}
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    title = response.choices[0].message.content.strip('"')
    print(f"   -> Title: {title}\n")
    return title


def main():
    """
    Runs a simple LLM chain that generates an idea, expands it, processes it,
    and creates a title. Each step is traced automatically.
    """
    print("🚀 Starting Simple OpenAI Tracing Example...")
    print("Each step will be captured as a span in your ZeroEval workspace.\n")

    # The main application logic, wrapped in a single parent span.
    with span(name="full_story_generation"):
        idea = generate_story_idea()
        paragraph = expand_on_idea(idea)
        character = extract_main_character(paragraph)
        title = write_final_title(paragraph, character)

    print("-" * 50)
    print("✅ Chain completed!")
    print(f"\nGenerated Story Title: {title}")
    print("\n🔍 Check your ZeroEval dashboard to see the full, nested trace.")


if __name__ == "__main__":
    main()
