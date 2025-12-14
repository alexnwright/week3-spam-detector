"""spam_detector.py
A simple spam email classifier using supervised learning (Decision Tree).

Run:
    python spam_detector.py

Dependencies:
    pandas, numpy, matplotlib, scikit-learn
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Keep sklearn warnings from cluttering output in some environments
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class SpamDetector:
    """A simple spam detector using a Decision Tree classifier."""

    def __init__(self, max_depth: int = 3, random_state: int = 42):
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.is_trained: bool = False
        self.data: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.split: Optional[SplitData] = None
        self.y_pred: Optional[np.ndarray] = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load email data from a CSV file."""
        self.data = pd.read_csv(filepath)
        print(f"Loaded {len(self.data)} emails from {filepath}")
        return self.data

    def prepare_features(self) -> None:
        """Separate features (X) from labels (y)."""
        if self.data is None:
            raise RuntimeError("No data loaded. Call load_data() first.")

        feature_columns = ["word_count", "exclamations", "money_words", "all_caps"]
        self.X = self.data[feature_columns]
        self.y = self.data["is_spam"]

        print("\n📊 Feature Statistics:")
        print("-" * 40)
        print("Average values by email type:")
        print(self.data.groupby("is_spam")[feature_columns].mean(numeric_only=True))

    def split_data(self, test_size: float = 0.3) -> None:
        """Split into training and testing sets."""
        if self.X is None or self.y is None:
            raise RuntimeError("Features not prepared. Call prepare_features() first.")

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
        )
        self.split = SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        print(f"\n📚 Training set: {len(X_train)} emails")
        print(f"🧪 Testing set: {len(X_test)} emails")

    def train(self) -> float:
        """Train the model on training data and return training accuracy."""
        if self.split is None:
            raise RuntimeError("Data not split. Call split_data() first.")

        print("\n🤖 Training the spam detector...")
        self.model.fit(self.split.X_train, self.split.y_train)
        self.is_trained = True

        train_pred = self.model.predict(self.split.X_train)
        train_accuracy = float(accuracy_score(self.split.y_train, train_pred))
        print(f"Training accuracy: {train_accuracy:.1%}")
        return train_accuracy

    def test(self) -> float:
        """Test the model on unseen data and return test accuracy."""
        if not self.is_trained or self.split is None:
            raise RuntimeError("Model not trained or data not split.")

        print("\n🧪 Testing on new emails...")
        self.y_pred = self.model.predict(self.split.X_test)
        test_accuracy = float(accuracy_score(self.split.y_test, self.y_pred))
        print(f"Testing accuracy: {test_accuracy:.1%}")

        cm = confusion_matrix(self.split.y_test, self.y_pred)
        print("\nConfusion Matrix:")
        print(" Predicted")
        print(" Normal Spam")
        print(f"Actual Normal {cm[0,0]:3d} {cm[0,1]:3d}")
        print(f"Actual Spam   {cm[1,0]:3d} {cm[1,1]:3d}")
        return test_accuracy

    def predict_email(self, word_count: int, exclamations: int, money_words: int, all_caps: int) -> int:
        """Predict if a single email is spam."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet!")

        features = np.array([[word_count, exclamations, money_words, all_caps]], dtype=float)
        prediction = int(self.model.predict(features)[0])
        probability = self.model.predict_proba(features)[0]

        result = "SPAM" if prediction == 1 else "NORMAL"
        confidence = float(np.max(probability) * 100)

        print("\n📧 Email Analysis:")
        print(f" Word count: {word_count}")
        print(f" Exclamation marks: {exclamations}")
        print(f" Money-related words: {money_words}")
        print(f" ALL CAPS words: {all_caps}")
        print(f"\n🎯 Prediction: {result}")
        print(f"🔍 Confidence: {confidence:.1f}%")
        return prediction

    def visualize_data(self, output_path: str = "spam_analysis.png") -> None:
        """Create visualizations to understand the data."""
        if self.data is None:
            raise RuntimeError("No data loaded.")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Word count distribution
        axes[0, 0].hist(
            [
                self.data[self.data["is_spam"] == 0]["word_count"],
                self.data[self.data["is_spam"] == 1]["word_count"],
            ],
            label=["Normal", "Spam"],
            alpha=0.7,
            bins=10,
        )
        axes[0, 0].set_xlabel("Word Count")
        axes[0, 0].set_ylabel("Number of Emails")
        axes[0, 0].set_title("Word Count Distribution")
        axes[0, 0].legend()

        # Plot 2: Exclamation marks
        axes[0, 1].bar(
            ["Normal", "Spam"],
            [
                self.data[self.data["is_spam"] == 0]["exclamations"].mean(),
                self.data[self.data["is_spam"] == 1]["exclamations"].mean(),
            ],
        )
        axes[0, 1].set_ylabel("Average Exclamation Marks")
        axes[0, 1].set_title("Exclamation Usage")

        # Plot 3: Money words
        axes[1, 0].bar(
            ["Normal", "Spam"],
            [
                self.data[self.data["is_spam"] == 0]["money_words"].mean(),
                self.data[self.data["is_spam"] == 1]["money_words"].mean(),
            ],
        )
        axes[1, 0].set_ylabel("Average Money Words")
        axes[1, 0].set_title("Money Word Usage")

        # Plot 4: Feature importance
        if self.is_trained:
            importance = self.model.feature_importances_
            features = ["Word Count", "Exclamations", "Money Words", "All Caps"]
            axes[1, 1].bar(features, importance)
            axes[1, 1].set_ylabel("Importance")
            axes[1, 1].set_title("What the Model Learned is Important")
            axes[1, 1].tick_params(axis="x", rotation=45)
        else:
            axes[1, 1].axis("off")
            axes[1, 1].set_title("Train the model to see feature importances")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.show()
        print(f"\n📊 Saved visualization as '{output_path}'")


def main() -> None:
    """Main program with interactive menu."""
    print("=" * 50)
    print("🚫 SPAM EMAIL DETECTOR")
    print("Machine Learning in Action!")
    print("=" * 50)

    detector = SpamDetector()

    # Create the data if it doesn't exist
    if not os.path.exists(os.path.join("data", "email_data.csv")):
        print("\n📝 Creating email dataset...")
        os.makedirs("data", exist_ok=True)
        from prepare_data import create_sample_emails

        data = create_sample_emails()
        data.to_csv(os.path.join("data", "email_data.csv"), index=False)

    # Load and prepare data
    detector.load_data(os.path.join("data", "email_data.csv"))
    detector.prepare_features()
    detector.split_data()

    # Train/test
    detector.train()
    detector.test()

    # Visualize
    print("\n📊 Creating visualizations...")
    detector.visualize_data()

    # Interactive testing
    while True:
        print("\n" + "=" * 50)
        print("TEST THE SPAM DETECTOR")
        print("=" * 50)
        print("1. Test a custom email")
        print("2. Test example spam email")
        print("3. Test example normal email")
        print("4. Show model performance")
        print("5. Learn about overfitting")
        print("6. Exit")

        choice = input("\nChoice (1-6): ").strip()

        if choice == "1":
            print("\nDescribe your email:")
            try:
                words = int(input("Approximate word count: "))
                exclaim = int(input("Number of exclamation marks: "))
                money = int(input("Money-related words (free, cash, win, etc.): "))
                caps = int(input("Words in ALL CAPS: "))
                detector.predict_email(words, exclaim, money, caps)
            except ValueError:
                print("Please enter numbers only!")
            except RuntimeError as e:
                print(str(e))

        elif choice == "2":
            print("\nTesting spam email: 'WIN FREE CASH NOW!!!'")
            detector.predict_email(30, 3, 4, 3)

        elif choice == "3":
            print("\nTesting normal email: 'Meeting tomorrow at 2pm'")
            detector.predict_email(100, 0, 0, 0)

        elif choice == "4":
            detector.test()

        elif choice == "5":
            print("\n📚 ABOUT OVERFITTING")
            print("-" * 40)
            print("Overfitting is when your model:")
            print("• Memorizes the training data perfectly")
            print("• But fails on new, unseen data")
            print("\nIt's like a student who memorizes answers")
            print("but doesn't understand the concepts!")
            print("\nWe prevent it by:")
            print("• Using separate test data")
            print("• Keeping our model simple")
            print("• Having enough diverse training examples")

        elif choice == "6":
            print("\nThanks for using Spam Detector!")
            break

        else:
            print("Please choose a number from 1 to 6.")


if __name__ == "__main__":
    main()
