#!/usr/bin/env python3
"""
Comedian A/B Testing Example
=============================

This example demonstrates a real-world A/B test comparing OpenAI models
for generating comedy material. We test which model produces funnier jokes
across different comedy segments, using ze.choose() for model selection with
timeboxed experiments and signals to track joke quality.

Features demonstrated:
- Timeboxed experiments with duration_days
- Signal tracking for joke quality (success/failure)
- Model comparison with performance metrics
- Nested spans for detailed tracing

Usage:
    python comedian_ab_test.py              # Run once
    python comedian_ab_test.py --runs 10    # Run 10 times to see distribution
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
import re

from dotenv import load_dotenv
from openai import OpenAI
import zeroeval as ze


def ensure_env():
    """Load and validate environment variables."""
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    # # PROD
    zeroeval_key = os.getenv("ZEROEVAL_API_KEY")
    zeroeval_api_url = os.getenv("ZEROEVAL_API_URL", "https://api.zeroeval.com")
    llm_base_url = os.getenv("ZEROEVAL_LLM_BASE_URL", "https://api.zeroeval.com/v1")

    if not zeroeval_key:
        raise RuntimeError(
            "Missing ZEROEVAL_API_KEY. Please set it in your .env file or environment."
        )

    return zeroeval_key, zeroeval_api_url, llm_base_url


COMEDY_SEGMENTS = [
    {
        "theme": "airline_food",
        "prompt": "Write a short, punchy standup bit about airline food. Make it observational and relatable.",
        "constraints": "Keep it PG-13, focus on universal experiences"
    },
    {
        "theme": "smartphones",
        "prompt": "Create a funny observation about our relationship with smartphones and social media.",
        "constraints": "Modern, relatable to all ages"
    },
    {
        "theme": "dating_apps",
        "prompt": "Tell a humorous story about the absurdity of modern dating apps.",
        "constraints": "Self-deprecating humor, PG-13"
    },
    {
        "theme": "working_from_home",
        "prompt": "Craft a comedic take on the chaos of working from home and video calls.",
        "constraints": "Post-pandemic humor, workplace appropriate"
    },
]

MODEL_VARIANTS = {
    "mini": "gpt-4.1-mini-2025-04-14",
    "advanced": "claude-opus-4-20250514",
}

MODEL_WEIGHTS = {
    "mini": 0.5,
    "advanced": 0.5,
}

TOUR_METADATA = {
    "tour_name": "ZeroEval Comedy Night",
    "venue": "The Algorithm Theater",
    "audience_mood": "tech-savvy",
}


def run_comedy_show(client, run_number=None):
    """Run a single comedy show and return results."""
    show_prefix = f"[Run {run_number}] " if run_number else ""
    
    print(f"\n{show_prefix}🎭 Welcome to ZeroEval Comedy Night!")
    print("=" * 60)
    
    # IMPORTANT: ze.choose() must be called within a span context
    # All signals attached to this span will be linked to the AB test variant
    with ze.span("comedy_show", tags=TOUR_METADATA) as show_span:
        # Make the AB test choice - this attaches ab_choice_id to show_span
        selected_model = ze.choose(
            name="comedy_model_advanced_vs_mini_v4",
            variants=MODEL_VARIANTS,
            weights=MODEL_WEIGHTS,
            duration_days=14,  # Run experiment for 2 weeks
            default_variant="mini"  # Use mini as fallback after experiment ends
        )
        
        # Find which variant key was selected for reporting
        variant_key = next(k for k, v in MODEL_VARIANTS.items() if v == selected_model)
        
        print(f"\n🎲 Model Selection: {variant_key} -> {selected_model}")
        print(f"   Testing {len(COMEDY_SEGMENTS)} comedy segments...\n")
        print("=" * 60)
        
        results = []
        
        for i, segment in enumerate(COMEDY_SEGMENTS, 1):
            print(f"\n📝 Segment {i}: {segment['theme'].replace('_', ' ').title()}")
            print("-" * 60)
            
            with ze.span(
                "joke_generation",
                tags={
                    "segment": segment["theme"],
                    "model_variant": variant_key,
                    "model_name": selected_model,
                    "segment_number": str(i)
                }
            ):
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a professional standup comedian. "
                                "Write tight, funny material that gets big laughs. "
                                "Keep it conversational and authentic."
                            )
                        },
                        {
                            "role": "user",
                            "content": f"{segment['prompt']}\n\nConstraints: {segment['constraints']}"
                        }
                    ],
                    temperature=1.0,
                    max_completion_tokens=256
                )
                
                joke = response.choices[0].message.content.strip()
                print(f"\n{joke}\n")
            
            with ze.span(
                "audience_feedback",
                tags={
                    "segment": segment["theme"],
                    "segment_number": str(i)
                }
            ) as feedback_span:
                feedback_response = client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an experienced comedy critic and audience member. "
                                "Rate jokes objectively based on originality, timing, "
                                "relatability, and laugh potential."
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Rate this comedy bit from 1-10 for live audience laughs:\n\n"
                                f"{joke}\n\n"
                                f"Respond in JSON format with 'rating' (number) and 'notes' (brief feedback)."
                            )
                        }
                    ],
                    temperature=1.0,
                    max_completion_tokens=150,
                    response_format={"type": "json_object"}
                )

                feedback_message = feedback_response.choices[0].message
                raw_content = feedback_message.content

                if isinstance(raw_content, list):
                    raw_content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in raw_content
                    ).strip()
                elif raw_content is None:
                    raw_content = ""
                else:
                    raw_content = str(raw_content).strip()

                feedback_data = {}
                if raw_content:
                    try:
                        feedback_data = json.loads(raw_content)
                    except json.JSONDecodeError:
                        match = re.search(r"(\d+(?:\.\d+)?)", raw_content)
                        if match:
                            feedback_data["rating"] = float(match.group(1))
                        feedback_data["notes"] = raw_content
                else:
                    feedback_data["notes"] = "No feedback returned"

                rating = float(feedback_data.get("rating", 0))
                notes = feedback_data.get("notes", "No feedback provided")

                # Track individual joke quality (optional - for detailed analysis)
                ze.set_signal(feedback_span, {
                    "joke_quality": rating >= 7.0,
                })

                print(f"🎯 Audience Rating: {rating}/10 {'✅' if rating >= 7.0 else '❌'}")
                print(f"💬 Notes: {notes}")
            
            results.append({
                "segment": segment["theme"],
                "joke": joke,
                "rating": rating,
                "notes": notes
            })
        
        # After all segments, attach aggregate signals to the show span
        # CRITICAL: These signals are attached to the same span where ze.choose() was called
        # This links them to the AB test variant via span_ab_choices table in the backend
        average_rating = sum(item["rating"] for item in results) / len(results)
        solid_jokes = sum(1 for r in results if r["rating"] >= 7.0)
        standout_jokes = sum(1 for r in results if r["rating"] >= 8.5)
        needs_revision = sum(1 for r in results if r["rating"] <= 5.5)

        # Attach signals to the comedy_show span (same level as the choice)
        # These will automatically appear in the AB test dashboard's signal distribution chart
        # For A/B testing, use a single primary success metric to get meaningful comparison
        ze.set_signal(show_span, {
            "show_quality": average_rating >= 7.0,  # Primary metric: good show
        })
        
        print("\n" + "=" * 60)
        print(f"{show_prefix}📊 SHOW SUMMARY")
        print("=" * 60)
        print(f"\n🤖 Model Tested: {variant_key} ({selected_model})")
        
        best_segment = max(results, key=lambda item: item["rating"])
        worst_segment = min(results, key=lambda item: item["rating"])
        
        print(f"\n📈 Performance Metrics:")
        print(f"   Average Rating: {average_rating:.2f}/10")
        print(f"   Best Segment: {best_segment['segment'].replace('_', ' ').title()} ({best_segment['rating']}/10)")
        print(f"   Weakest Segment: {worst_segment['segment'].replace('_', ' ').title()} ({worst_segment['rating']}/10)")
        print(f"   Solid Laughs: {solid_jokes}/{len(results)} segments (rating ≥ 7.0)")
        print(f"   Standout Bits: {standout_jokes}/{len(results)} segments (rating ≥ 8.5)")
        print(f"   Needs Revision: {needs_revision}/{len(results)} segments (rating ≤ 5.5)")

        print(f"\n🎭 Segment Breakdown:")
        for item in results:
            segment_name = item['segment'].replace('_', ' ').title()
            print(f"   • {segment_name}: {item['rating']}/10")
        
        # Display tracked signals
        success_rate = (solid_jokes / len(results)) * 100
        
        print(f"\n🎯 Signal Tracked (attached to comedy_show span):")
        print(f"   ✓ show_quality: {average_rating >= 7.0} (average rating: {average_rating:.1f}/10)")
        print(f"\n   📊 Success Rate: {success_rate:.1f}% ({solid_jokes}/{len(results)} segments rated ≥7.0)")
        print(f"   💡 This signal is automatically linked to the AB test variant!")
        print(f"   💡 View variant performance comparison in the ZeroEval dashboard")
        
        print("\n" + "=" * 60)
        print(f"{show_prefix}✅ Comedy show complete!")
        print("=" * 60)
    
    return {
        "variant_key": variant_key,
        "model": selected_model,
        "average_rating": average_rating,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run comedian A/B testing with different OpenAI models"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to run the comedy show (default: 1)"
    )
    args = parser.parse_args()
    
    zeroeval_key, zeroeval_api_url, llm_base_url = ensure_env()
    ze.init(api_key=zeroeval_key, api_url=zeroeval_api_url, debug=True)
    client = OpenAI(api_key=zeroeval_key, base_url=llm_base_url)
    
    if args.runs == 1:
        run_comedy_show(client)
    else:
        print(f"\n🎪 Running {args.runs} comedy shows to analyze A/B test distribution...")
        print("=" * 80)
        
        all_results = []
        model_counts = defaultdict(int)
        model_ratings = defaultdict(list)
        
        for i in range(1, args.runs + 1):
            result = run_comedy_show(client, run_number=i)
            all_results.append(result)
            model_counts[result["variant_key"]] += 1
            model_ratings[result["variant_key"]].append(result["average_rating"])
        
        print("\n" + "=" * 80)
        print("📊 AGGREGATE ANALYSIS ACROSS ALL RUNS")
        print("=" * 80)
        
        print(f"\n🎲 Model Selection Distribution (n={args.runs}):")
        for variant_key in sorted(model_counts.keys()):
            count = model_counts[variant_key]
            percentage = (count / args.runs) * 100
            model_name = MODEL_VARIANTS[variant_key]
            expected_pct = MODEL_WEIGHTS[variant_key] * 100
            
            bar_length = int(percentage / 2)
            bar = "█" * bar_length
            
            print(f"   {variant_key:8} ({model_name:15}): {count:3}/{args.runs} ({percentage:5.1f}%) [Expected: {expected_pct:4.1f}%]")
            print(f"            {bar}")
        
        print(f"\n📈 Average Performance by Model:")
        for variant_key in sorted(model_ratings.keys()):
            ratings = model_ratings[variant_key]
            avg = sum(ratings) / len(ratings)
            min_rating = min(ratings)
            max_rating = max(ratings)
            model_name = MODEL_VARIANTS[variant_key]
            
            print(f"   {variant_key:8} ({model_name:15}): {avg:.2f}/10 (min: {min_rating:.2f}, max: {max_rating:.2f}, n={len(ratings)})")
        
        overall_avg = sum(r["average_rating"] for r in all_results) / len(all_results)
        print(f"\n🎭 Overall Average Rating: {overall_avg:.2f}/10 across {args.runs} shows")
        
        print("\n" + "=" * 80)
        print("✅ All comedy shows complete!")
        print("\n📊 View in ZeroEval Dashboard:")
        print("   • Navigate to: Monitoring → A/B Testing → comedy_model_advanced_vs_mini")
        print("   • Variant Performance: Compare 'show_quality' success rates between models")
        print("   • Signal Impact: See how each variant performs on the quality metric")
        print("   • Track which model consistently generates better-rated comedy")
        print("=" * 80)


if __name__ == "__main__":
    main()

