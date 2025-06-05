import zeroeval as ze
import time
from zeroeval.observability.decorators import span
from zeroeval.observability.tracer import tracer
import random
import openai

# Configure tracer
tracer.configure(
    flush_interval=1.0,
    max_spans=50
)

# Initialize ZeroEval
ze.init(api_key="sk_ze_4OxO2q-uR6beq32qxV-zPkq0uONq4CIjtS_Bc7P9idM")

# Pull the Email-Sentiment dataset
dataset = ze.Dataset.pull("Email-Sentiment")

@span(name="email_sentiment_task")
def email_sentiment_task(row):
    """
    This task predicts the sentiment of an email using the OpenAI API.
    """
    input_email = row["input"]
    prediction = predict_sentiment(input_email)
    return prediction

@span(name="predict_sentiment")
def predict_sentiment(email_text):
    """
    Uses OpenAI to predict the sentiment of an email.
    """
    client = openai.OpenAI(api_key="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")
    
    prompt = f"""
    Analyze the sentiment of the following email and classify it as either 'positive', 'neutral', or 'negative'.
    Respond with just one word: 'positive', 'neutral', or 'negative'.
    
    Email: {email_text}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Extract the predicted sentiment
    prediction = response.choices[0].message.content.strip().lower()
    
    # Ensure the prediction is one of the valid classes
    if prediction not in ["positive", "neutral", "negative"]:
        # Default to neutral if the model gives an unexpected response
        prediction = "neutral"
    
    time.sleep(random.uniform(0.1, 0.3))  # Simulate processing time
    
    return prediction

@span(name="sentiment_accuracy_evaluator")
def sentiment_accuracy_evaluator(row, output):
    """
    Evaluates if the predicted sentiment matches the ground truth.
    Returns 1.0 for correct predictions, 0.0 for incorrect ones.
    """
    ground_truth = row["output"]
    prediction = output
    
    # Check if prediction matches ground truth
    is_correct = prediction.lower() == ground_truth.lower()
    
    return 1.0 if is_correct else 0.0

@span(name="sentiment_confidence_evaluator")
def sentiment_confidence_evaluator(row, output):
    """
    Evaluates the confidence of the sentiment prediction.
    Uses a simulated confidence score between 0.5 and 1.0.
    """
    ground_truth = row["output"]
    prediction = output
    
    # Base confidence score - higher for correct predictions
    if prediction.lower() == ground_truth.lower():
        # For correct predictions, confidence between 0.7 and 1.0
        confidence = 0.7 + (random.random() * 0.3)
    else:
        # For incorrect predictions, confidence between 0.5 and 0.7
        confidence = 0.5 + (random.random() * 0.2)
    
    return confidence

# Create the experiment
experiment = ze.Experiment(
    dataset=dataset,
    task=email_sentiment_task,
    evaluators=[
        sentiment_accuracy_evaluator,
        sentiment_confidence_evaluator
    ]
)

# Run the experiment
if __name__ == "__main__":
    print("Starting Email Sentiment Analysis Experiment...")
    experiment.run()
    print("Experiment completed!") 