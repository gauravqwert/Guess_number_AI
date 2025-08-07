import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier
import joblib


def generate_dataset(num_samples=10000, max_number=100):
    data = []
    for _ in range(num_samples):
        target = random.randint(1, max_number)
        guesses = []
        feedback = []
        low, high = 1, max_number

        while True:
            if not guesses:
                guess = max_number // 2
            else:
                last_feedback = feedback[-1]
                if last_feedback == "higher":
                    low = max(low, guesses[-1] + 1)
                elif last_feedback == "lower":
                    high = min(high, guesses[-1] - 1)

                guess = random.randint(low, high) if low <= high else random.randint(1, max_number)

            if guess == target:
                feedback.append('correct')
                break
            elif guess < target:
                feedback.append('higher')
            else:
                feedback.append('lower')
            guesses.append(guess)

        for i in range(1, len(guesses)):
            current_low = 1
            current_high = max_number
            for j in range(i):
                if feedback[j] == "higher":
                    current_low = max(current_low, guesses[j] + 1)
                elif feedback[j] == "lower":
                    current_high = min(current_high, guesses[j] - 1)

            row = {
                'current_guess': guesses[i - 1],
                'previous_guess': guesses[i - 2] if i > 1 else 0,
                'guess_count': i,
                'low_bound': current_low,
                'high_bound': current_high,
                'last_feedback': 0 if feedback[i - 1] == "lower" else 1,
                'proposed_guess': guesses[i],
                'next_feedback': feedback[i]
            }
            data.append(row)

    return pd.DataFrame(data)


# Generate and save dataset
print("Generating dataset...")
df = generate_dataset(num_samples=5000, max_number=100)

# Map feedback to numerical values
feedback_map = {'lower': 0, 'higher': 1, 'correct': 2}
df['next_feedback_num'] = df['next_feedback'].map(feedback_map)

# Select features and target
features = ['current_guess', 'previous_guess', 'guess_count',
            'low_bound', 'high_bound', 'last_feedback', 'proposed_guess']
X = df[features]
y = df['next_feedback_num']

# Train and save model
print("Training model...")
model = DecisionTreeClassifier(max_depth=8, min_samples_split=5, random_state=42)
model.fit(X, y)

joblib.dump(model, 'number_guesser_model.pkl')
print("Model saved as 'number_guesser_model.pkl'")