import streamlit as st
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Streamlit UI
st.title("Tweet Sentiment Analyzer 🐦")
st.write("Tweet type chey, adi **Positive**, **Negative**, or **Neutral** ani cheptha!")

# User input
tweet_input = st.text_area("Tweet type chey:", height=100)

if st.button("Analyze Cheyyi"):
    if not tweet_input.strip():
        st.error("Tweet type cheyyali, empty ga vaddu!")
    else:
        # Sentiment analysis
        scores = analyzer.polarity_scores(tweet_input)
        
        # Determine sentiment
        compound_score = scores['compound']
        if compound_score >= 0.05:
            sentiment = "Positive"
        elif compound_score <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        # Display result
        st.success(f"**Sentiment**: {sentiment}")
        st.write(f"**Tweet**: {tweet_input}")
        
        # Display sentiment scores
        st.subheader("Sentiment Scores")
        st.write(f"Positive: {scores['pos']:.2%}")
        st.write(f"Negative: {scores['neg']:.2%}")
        st.write(f"Neutral: {scores['neu']:.2%}")
        st.write(f"Compound: {scores['compound']:.2f}")
        
        # Bar chart for scores
        st.subheader("Sentiment Scores Visualization")
        labels = ['Positive', 'Negative', 'Neutral']
        values = [scores['pos'], scores['neg'], scores['neu']]
        fig, ax = plt.subplots()
        ax.bar(labels, values, color=['green', 'red', 'blue'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.set_title('Sentiment Distribution')
        st.pyplot(fig)
