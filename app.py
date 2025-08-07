import streamlit as st
import joblib
import random
import pandas as pd

# Load the model
model = joblib.load('number_guesser_model.pkl')

# App layout
st.set_page_config(page_title="AI Number Guesser", page_icon="🔢", layout="centered")
st.title("🔢 AI Number Guesser")
st.write("Watch the AI try to guess your secret number!")

# Initialize session state
if 'target' not in st.session_state:
    st.session_state.target = random.randint(1, 100)
    st.session_state.guesses = []
    st.session_state.feedback = []
    st.session_state.low = 1
    st.session_state.high = 100
    st.session_state.game_over = False

# Sidebar controls
with st.sidebar:
    st.header("Game Settings")
    max_number = st.slider("Maximum number", 10, 1000, 100)
    if st.button("New Game"):
        st.session_state.target = random.randint(1, max_number)
        st.session_state.guesses = []
        st.session_state.feedback = []
        st.session_state.low = 1
        st.session_state.high = max_number
        st.session_state.game_over = False
        st.rerun()  # Changed from experimental_rerun()


# Game logic
def make_ai_guess():
    if not st.session_state.guesses:
        # First guess
        new_guess = max_number // 2
    else:
        # Prepare features
        features = {
            'current_guess': st.session_state.guesses[-1],
            'previous_guess': st.session_state.guesses[-2] if len(st.session_state.guesses) > 1 else 0,
            'guess_count': len(st.session_state.guesses),
            'low_bound': st.session_state.low,
            'high_bound': st.session_state.high,
            'last_feedback': 0 if st.session_state.feedback[-1] == "lower" else 1,
            'proposed_guess': (st.session_state.low + st.session_state.high) // 2
        }

        # Generate candidate guesses
        candidates = [
            st.session_state.low,
            (st.session_state.low + st.session_state.high) // 2,
            st.session_state.high,
            st.session_state.guesses[-1] + 1,
            st.session_state.guesses[-1] - 1
        ]
        candidates = [c for c in candidates if st.session_state.low <= c <= st.session_state.high]
        candidates = list(set(candidates))  # Remove duplicates

        # Predict best candidate
        best_guess = candidates[0]
        best_score = -1
        for candidate in candidates:
            current_features = features.copy()
            current_features['proposed_guess'] = candidate
            features_df = pd.DataFrame([current_features])
            try:
                pred_proba = model.predict_proba(features_df)[0]
                score = pred_proba[2] if len(pred_proba) > 2 else 0
            except:
                score = 0

            if score > best_score:
                best_score = score
                best_guess = candidate

        new_guess = best_guess

    # Check guess
    if new_guess == st.session_state.target:
        st.session_state.game_over = True
        feedback = "correct"
    elif new_guess < st.session_state.target:
        feedback = "higher"
        st.session_state.low = new_guess + 1
    else:
        feedback = "lower"
        st.session_state.high = new_guess - 1

    st.session_state.guesses.append(new_guess)
    st.session_state.feedback.append(feedback)


# Game display
if not st.session_state.game_over:
    if st.button("Make AI Guess"):
        make_ai_guess()
        st.rerun()  # Changed from experimental_rerun()
else:
    st.success(f"🎉 AI guessed your number {st.session_state.target} in {len(st.session_state.guesses)} tries!")

    # Show guess history
    st.subheader("Guess History")
    history_df = pd.DataFrame({
        "Guess": st.session_state.guesses,
        "Feedback": st.session_state.feedback
    })
    st.dataframe(history_df.style.applymap(
        lambda x: 'background-color: #c8e6c9' if x == 'correct' else '',
        subset=['Feedback']
    ))

# Current game info
st.write("---")
col1, col2 = st.columns(2)
with col1:
    st.metric("Target Number", "?" if not st.session_state.game_over else st.session_state.target)
with col2:
    st.metric("Guesses Made", len(st.session_state.guesses))

# How to play
with st.expander("How to Play"):
    st.write("""
    1. Set the maximum number range using the slider
    2. Click 'New Game' to start
    3. Click 'Make AI Guess' to see the AI's next guess
    4. The AI will use its model to narrow down the number
    5. Watch as it tries to guess your secret number!
    """)