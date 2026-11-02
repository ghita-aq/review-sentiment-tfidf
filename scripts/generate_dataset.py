"""
Generate a synthetic sentiment dataset for training and testing.

Creates a CSV file with labeled customer reviews (positive, negative, neutral)
to be used when a real dataset is not available. The generated dataset contains
a balanced mix of sentiments with realistic review text patterns.

Usage:
    python generate_dataset.py [--output data/reviews.csv] [--num-samples 2000]
"""

import csv
import argparse
import random

# ---------------------------------------------------------------------------
# Review templates by sentiment
# ---------------------------------------------------------------------------

POSITIVE_TEMPLATES = [
    "This product is absolutely {adj}! I {verb} it and would {verb2} to anyone.",
    "I am so {adj} with this purchase. The quality is {adj2} and exceeded my expectations.",
    "{adj} product! Best purchase I've ever made. Customer service was {adj2}.",
    "I {verb} this product! It works perfectly and the build quality is {adj}.",
    "Outstanding quality and {adj} performance. I'm completely {adj2} with this purchase.",
    "This is exactly what I was looking for. {adj} design and {adj2} functionality.",
    "Highly recommend this product. It's {adj}, reliable, and {adj2}.",
    "I've been using this for weeks now and I'm {adj}. It's truly {adj2}.",
    "Five stars! This product is {adj} and the customer service was {adj2}.",
    "Wonderful experience from start to finish. The product is {adj} and {adj2}.",
    "I'm impressed with the quality. This is a {adj} product that works {adj2}.",
    "Great value for money. The product is {adj} and performs {adj2}.",
    "This exceeded all my expectations. Truly {adj} and {adj2} in every way.",
    "I bought this as a gift and the recipient {verb} it. {adj} quality!",
    "Perfect in every way. {adj} materials, {adj2} craftsmanship, and fast delivery.",
    "I can't say enough good things about this product. It's simply {adj}.",
    "The best product in its category. {adj} features and {adj2} build quality.",
    "I'm a happy customer. This product is {adj} and works exactly as described.",
    "Fantastic product! {adj} quality and {adj2} performance. Will buy again.",
    "This is a must-have product. {adj} design and {adj2} functionality.",
]

NEGATIVE_TEMPLATES = [
    "This product is absolutely {adj}. I {verb} it and would not {verb2} it to anyone.",
    "I am so {adj} with this purchase. The quality is {adj2} and below expectations.",
    "{adj} product. Worst purchase I've ever made. Customer service was {adj2}.",
    "I {verb} this product. It broke after one day and the build quality is {adj}.",
    "Terrible quality and {adj} performance. I'm completely {adj2} with this purchase.",
    "This is not what I ordered. {adj} design and {adj2} functionality.",
    "Do not buy this product. It's {adj}, unreliable, and {adj2}.",
    "I've been using this for a week and I'm {adj}. It's truly {adj2}.",
    "One star. This product is {adj} and the customer service was {adj2}.",
    "Horrible experience from start to finish. The product is {adj} and {adj2}.",
    "I'm disappointed with the quality. This is a {adj} product that doesn't work {adj2}.",
    "Complete waste of money. The product is {adj} and performs {adj2}.",
    "This fell far short of expectations. Truly {adj} and {adj2} in every way.",
    "I bought this and immediately {verb} it. {adj} quality!",
    "Defective product. {adj} materials, {adj2} craftsmanship, and slow delivery.",
    "I can't say anything good about this product. It's simply {adj}.",
    "The worst product in its category. {adj} features and {adj2} build quality.",
    "I'm a very unhappy customer. This product is {adj} and doesn't work as described.",
    "Awful product! {adj} quality and {adj2} performance. Will never buy again.",
    "Stay away from this product. {adj} design and {adj2} functionality.",
]

NEUTRAL_TEMPLATES = [
    "This product is {adj}. It does what it's supposed to do, nothing more.",
    "I'm {adj} with this purchase. The quality is {adj2} for the price.",
    "{adj} product. It's okay, not great but not terrible either.",
    "The product is {adj}. It works as expected but nothing {adj2}.",
    "Average quality and {adj} performance. It's a {adj2} product overall.",
    "This is a {adj} product. It meets basic expectations but isn't {adj2}.",
    "It's an {adj} product for the price. Nothing {adj2} but gets the job done.",
    "I've been using this for a while. It's {adj} and {adj2} at best.",
    "Three stars. This product is {adj} and the customer service was {adj2}.",
    "Decent experience overall. The product is {adj} and {adj2}.",
    "The quality is {adj}. It's a {adj2} product that works for basic needs.",
    "Fair value for money. The product is {adj} and performs {adj2}.",
    "This met my basic expectations. It's {adj} and {adj2} enough.",
    "I bought this for everyday use. It's {adj} and {adj2} quality.",
    "It's an {adj} product. {adj2} materials and average craftsmanship.",
    "Nothing special about this product. It's {adj} and does the basics.",
    "Middle of the road product. {adj} features and {adj2} build quality.",
    "It's an {adj} purchase. The product is {adj2} and works as described.",
    "Okay product. {adj} quality and {adj2} performance. Neither good nor bad.",
    "This is a {adj} product. {adj2} design and average functionality.",
]

POSITIVE_ADJS = [
    "amazing", "wonderful", "excellent", "fantastic", "outstanding",
    "incredible", "superb", "brilliant", "perfect", "great",
    "impressive", "delightful", "remarkable", "exceptional", "top-notch",
]
POSITIVE_VERBS = ["love", "adore", "enjoy", "appreciate", "recommend"]
POSITIVE_VERBS2 = ["recommend", "suggest", "advise", "endorse"]
POSITIVE_ADJS2 = [
    "reliable", "durable", "smooth", "efficient", "responsive",
    "sturdy", "elegant", "sleek", "intuitive", "powerful",
]

NEGATIVE_ADJS = [
    "terrible", "awful", "horrible", "dreadful", "disappointing",
    "poor", "mediocre", "defective", "broken", "useless",
    "frustrating", "annoying", "unreliable", "flimsy", "cheap",
]
NEGATIVE_VERBS = ["hate", "despise", "regret", "dislike", "avoid"]
NEGATIVE_VERBS2 = ["recommend", "suggest", "advise", "endorse"]
NEGATIVE_ADJS2 = [
    "unreliable", "flimsy", "clunky", "inefficient", "unresponsive",
    "fragile", "ugly", "bulky", "confusing", "weak",
]

NEUTRAL_ADJS = [
    "okay", "average", "decent", "fair", "acceptable",
    "moderate", "standard", "ordinary", "typical", "adequate",
]
NEUTRAL_ADJS2 = [
    "reasonable", "passable", "satisfactory", "functional", "workable",
    "adequate", "sufficient", "tolerable", "manageable", "plain",
]


def fill_template(template: str, adj_pool, verb_pool, verb2_pool, adj2_pool) -> str:
    """Fill a template with random words from the given pools."""
    kwargs = {"adj": random.choice(adj_pool), "adj2": random.choice(adj2_pool)}
    if verb_pool is not None:
        kwargs["verb"] = random.choice(verb_pool)
    if verb2_pool is not None:
        kwargs["verb2"] = random.choice(verb2_pool)
    return template.format(**kwargs)


def generate_reviews(num_samples: int = 2000) -> list:
    """Generate a balanced dataset of labeled reviews."""
    reviews = []
    per_class = num_samples // 3

    # Positive reviews
    for _ in range(per_class):
        template = random.choice(POSITIVE_TEMPLATES)
        text = fill_template(template, POSITIVE_ADJS, POSITIVE_VERBS, POSITIVE_VERBS2, POSITIVE_ADJS2)
        reviews.append({"review": text, "sentiment": "positive"})

    # Negative reviews
    for _ in range(per_class):
        template = random.choice(NEGATIVE_TEMPLATES)
        text = fill_template(template, NEGATIVE_ADJS, NEGATIVE_VERBS, NEGATIVE_VERBS2, NEGATIVE_ADJS2)
        reviews.append({"review": text, "sentiment": "negative"})

    # Neutral reviews
    for _ in range(per_class):
        template = random.choice(NEUTRAL_TEMPLATES)
        text = fill_template(template, NEUTRAL_ADJS, None, None, NEUTRAL_ADJS2)
        reviews.append({"review": text, "sentiment": "neutral"})

    random.shuffle(reviews)
    return reviews


def save_to_csv(reviews: list, output_path: str) -> None:
    """Save reviews to a CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["review", "sentiment"])
        writer.writeheader()
        writer.writerows(reviews)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sentiment dataset")
    parser.add_argument(
        "--output",
        default="data/reviews.csv",
        help="Output CSV file path (default: data/reviews.csv)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="Number of samples to generate (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    reviews = generate_reviews(args.num_samples)
    save_to_csv(reviews, args.output)

    print(f"Generated {len(reviews)} reviews ({len(reviews) // 3} per class)")
    print(f"Saved to: {args.output}")
    print(f"  Positive: {sum(1 for r in reviews if r['sentiment'] == 'positive')}")
    print(f"  Negative: {sum(1 for r in reviews if r['sentiment'] == 'negative')}")
    print(f"  Neutral:  {sum(1 for r in reviews if r['sentiment'] == 'neutral')}")


if __name__ == "__main__":
    main()
