import json
from typing import List, Dict, Any
import zeroeval as ze

# Dataset: Email sentiment classification
# Input: Email body text
# Output: Sentiment classification (positive, neutral, negative)

ze.init(api_key="sk_ze_rDMKmDkyHuc_OXykkWUuwqUtlGNx2auUgNifP5THobw")

dataset = ze.Dataset(
    name="Email-Sentiment",
    description="A dataset for classifying email sentiment as positive, neutral, or negative based on email body text",
    data=[
        {
            "input": "I am extremely disappointed with your customer service. I've been waiting for a response for over a week now, and no one has gotten back to me. This is unacceptable.",
            "output": "negative"
        },
        {
            "input": "Thank you so much for your quick response! The solution you provided fixed my issue immediately. I really appreciate your help.",
            "output": "positive"
        },
        {
            "input": "Just following up on my previous email regarding the order status. Could you please let me know when I can expect delivery?",
            "output": "neutral"
        },
        {
            "input": "I wanted to express my gratitude for the exceptional service I received from your team. Everyone was professional, courteous, and went above and beyond my expectations. I will definitely recommend your company to others!",
            "output": "positive"
        },
        {
            "input": "This is the third time I've had to contact you about this same issue. Your product continues to malfunction, and your support team hasn't provided any real solutions. I want a refund immediately.",
            "output": "negative"
        },
        {
            "input": "Please find attached the documents you requested for processing my application.",
            "output": "neutral"
        },
        {
            "input": "I'm writing to confirm that I've received your package. Everything seems to be in order.",
            "output": "neutral"
        },
        {
            "input": "Your team's performance has exceeded all our expectations. The project was delivered ahead of schedule and the quality of work is outstanding. We look forward to continuing our partnership.",
            "output": "positive"
        },
        {
            "input": "I regret to inform you that we will be canceling our subscription. The service quality has declined significantly over the past few months and is no longer worth the cost.",
            "output": "negative"
        },
        {
            "input": "Could you please provide me with more information about your premium services? I'm interested in learning about the additional features.",
            "output": "neutral"
        },
        {
            "input": "I'm absolutely thrilled with my recent purchase! The product is even better than I expected, and it arrived two days earlier than the estimated delivery date. You've earned a loyal customer!",
            "output": "positive"
        },
        {
            "input": "This is outrageous! I was charged twice for my order and despite multiple attempts to contact your billing department, I've received no assistance. If this isn't resolved immediately, I'll be forced to dispute the charge with my bank.",
            "output": "negative"
        },
        {
            "input": "I'm writing to request a copy of my invoice for order #12345 for my records.",
            "output": "neutral"
        },
        {
            "input": "Just wanted to say that your webinar yesterday was incredibly informative. I learned so much and can't wait to implement these strategies in my own work. Thank you for sharing your expertise!",
            "output": "positive"
        },
        {
            "input": "The quality of your product is terrible. It broke after just one week of normal use. This is clearly not the premium item that was advertised. I feel completely misled by your marketing.",
            "output": "negative"
        },
        {
            "input": "I'll be out of the office next week. Please direct any urgent matters to my colleague, John Smith, at john.smith@example.com.",
            "output": "neutral"
        },
        {
            "input": "I can't express how happy I am with the results of our collaboration. Your team's creativity and dedication have transformed our brand. The feedback from our customers has been overwhelmingly positive.",
            "output": "positive"
        },
        {
            "input": "I've been a loyal customer for five years, but after this experience, I'm taking my business elsewhere. Your company clearly doesn't value customer loyalty or satisfaction anymore.",
            "output": "negative"
        },
        {
            "input": "Please note that our office will be closed on Monday, May 30th in observance of the holiday. We will resume normal business hours on Tuesday, May 31st.",
            "output": "neutral"
        },
        {
            "input": "Wow! I just tried your new feature and it's absolutely game-changing. It's solved a problem I've been struggling with for months. Brilliant work by your development team!",
            "output": "positive"
        },
        {
            "input": "Your shipping costs are ridiculous and your return policy is even worse. I had to pay nearly half the product cost just to return a defective item. This is the worst online shopping experience I've ever had.",
            "output": "negative"
        },
        {
            "input": "I'm inquiring about the status of my application submitted on April 15th. Has it been processed yet?",
            "output": "neutral"
        },
        {
            "input": "I want to commend your tech support team, especially Sarah, for the outstanding assistance provided yesterday. She was patient, knowledgeable, and stayed on the call until my complex issue was completely resolved. This level of service is rare these days!",
            "output": "positive"
        },
        {
            "input": "I've been trying to access my account for three days now and keep getting error messages. This is incredibly frustrating, and your help articles haven't addressed my specific issue at all. I need this resolved ASAP as I'm missing important deadlines.",
            "output": "negative"
        }
    ]
)

# Push the dataset to ZeroEval
dataset.push(create_new_version=True)