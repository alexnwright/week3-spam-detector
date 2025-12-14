"""prepare_data.py
Prepare email data for spam detection.

This script creates a small synthetic dataset of email 'features' and labels (spam vs normal)
and writes it to data/email_data.csv.

Run:
    python prepare_data.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


def create_sample_emails(seed: int = 42) -> pd.DataFrame:
    """Create and return a simple dataset of emails.

    In real life, we'd use thousands of real emails; here we create a small,
    hand-crafted dataset and expand it with slight random variations.

    Columns:
        subject (str)
        word_count (int)
        exclamations (int)
        money_words (int)
        all_caps (int)
        is_spam (int; 1 spam, 0 normal)
    """
    rng = np.random.default_rng(seed)

    emails = [
        # Spam emails (label = 1)
        {
            "subject": "WIN MONEY NOW!!!",
            "word_count": 50,
            "exclamations": 3,
            "money_words": 5,
            "all_caps": 3,
            "is_spam": 1,
        },
        {
            "subject": "Congratulations! You've won!",
            "word_count": 45,
            "exclamations": 2,
            "money_words": 4,
            "all_caps": 1,
            "is_spam": 1,
        },
        {
            "subject": "FREE OFFER LIMITED TIME",
            "word_count": 30,
            "exclamations": 1,
            "money_words": 3,
            "all_caps": 4,
            "is_spam": 1,
        },
        {
            "subject": "Make $$$ working from home!",
            "word_count": 40,
            "exclamations": 1,
            "money_words": 6,
            "all_caps": 0,
            "is_spam": 1,
        },
        {
            "subject": "URGENT: Claim your prize",
            "word_count": 35,
            "exclamations": 0,
            "money_words": 2,
            "all_caps": 1,
            "is_spam": 1,
        },
        {
            "subject": "Discount pharmacy prices!!!",
            "word_count": 25,
            "exclamations": 3,
            "money_words": 2,
            "all_caps": 0,
            "is_spam": 1,
        },
        {
            "subject": "You're a WINNER!",
            "word_count": 20,
            "exclamations": 1,
            "money_words": 1,
            "all_caps": 1,
            "is_spam": 1,
        },
        {
            "subject": "Get rich quick scheme",
            "word_count": 55,
            "exclamations": 0,
            "money_words": 4,
            "all_caps": 0,
            "is_spam": 1,
        },
        # Normal emails (label = 0)
        {
            "subject": "Meeting tomorrow at 3pm",
            "word_count": 120,
            "exclamations": 0,
            "money_words": 0,
            "all_caps": 0,
            "is_spam": 0,
        },
        {
            "subject": "Re: Project update",
            "word_count": 200,
            "exclamations": 0,
            "money_words": 0,
            "all_caps": 0,
            "is_spam": 0,
        },
        {
            "subject": "Lunch plans?",
            "word_count": 85,
            "exclamations": 1,
            "money_words": 0,
            "all_caps": 0,
            "is_spam": 0,
        },
        {
            "subject": "Your package has shipped",
            "word_count": 100,
            "exclamations": 0,
            "money_words": 0,
            "all_caps": 0,
            "is_spam": 0,
        },
        {
            "subject": "Happy Birthday!",
            "word_count": 150,
            "exclamations": 1,
            "money_words": 0,
            "all_caps": 0,
            "is_spam": 0,
        },
        {
            "subject": "Class cancelled today",
            "word_count": 75,
            "exclamations": 0,
            "money_words": 0,
            "all_caps": 0,
            "is_spam": 0,
        },
        {
            "subject": "Thanks for your help",
            "word_count": 90,
            "exclamations": 0,
            "money_words": 0,
            "all_caps": 0,
            "is_spam": 0,
        },
        {
            "subject": "See you tonight",
            "word_count": 60,
            "exclamations": 0,
            "money_words": 0,
            "all_caps": 0,
            "is_spam": 0,
        },
    ]

    df = pd.DataFrame(emails)

    # Expand dataset by creating variations
    more_samples = []
    for _, email in df.iterrows():
        for _i in range(3):
            new_email = email.copy()

            # Add random variation (clamped where needed)
            new_wc = int(new_email["word_count"]) + int(rng.integers(-10, 10))
            new_ex = int(new_email["exclamations"]) + int(rng.integers(-1, 2))
            new_ex = max(0, new_ex)

            new_email["word_count"] = max(1, new_wc)
            new_email["exclamations"] = new_ex

            more_samples.append(new_email)

    df_expanded = pd.concat([df, pd.DataFrame(more_samples)], ignore_index=True)
    return df_expanded


def main() -> None:
    os.makedirs("data", exist_ok=True)
    data = create_sample_emails()
    out_path = os.path.join("data", "email_data.csv")
    data.to_csv(out_path, index=False)

    print(f"Created dataset with {len(data)} emails")
    print(f"Spam emails: {len(data[data['is_spam'] == 1])}")
    print(f"Normal emails: {len(data[data['is_spam'] == 0])}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
