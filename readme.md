# NLP: Sentiment and Thematic Analysis of Spotify Reviews

## Introduction: Task Description

The goal of this task is to analyze a dataset of user reviews to understand both the overall sentiment and the main themes expressed. The process includes several steps:

1. **Preprocess data:** Clean and prepare the data for analysis.  
2. **Sentiment analysis:** Classify reviews as positive, negative, or neutral.  
3. **Thematic analysis:** Extract keywords or n-grams and group them into 3–5 main themes.  
4. **Visualize results:** Display the sentiment distribution and themes effectively.

---

## Dataset

- I initially planned to scrape data from a movie rating website, but it didn’t work for me. The site required repeatedly clicking a “Load More” button, and my Selenium setup wasn’t functioning as expected (a tool used to automate web browsers for tasks like web scraping and testing).  
- I then decided to use a dataset from Kaggle: Spotify reviews from the Google Play Store.  

The dataset contains the following columns:  

- **Time_submitted** – when the review was posted.  
- **Review** – the actual text of the user review.  
- **Rating** – user rating (1–5 stars).  
- **Total_thumbsup** – number of “helpful” votes.  
- **Reply** – developer reply (mostly empty/NaN).  

For the analysis (sentiment and themes), the **Review** column was the most important.

---

## Data Cleaning and Preparation

Before running sentiment and thematic analysis, the text needed preprocessing to remove noise and make it machine-readable. Steps followed:

1. **Lowercasing**  
   - Tool: Python string methods (`.lower()`)  
   - Why: Standardizes text (so “Great” and “great” are treated the same)  
   - What it does: Converts all words into lowercase  

2. **Removing Punctuation & Special Characters**  
   - Tool: Python `re` (regular expressions)  
   - Why: Punctuation and symbols add little meaning for sentiment/theme extraction  
   - What it does: Keeps only letters and spaces, removes things like `. , ! ? "`  

3. **Sentence Segmentation**  
   - Tool: spaCy (`doc.sents`)  
   - Why: Splits a long review into individual sentences for sentence-level analysis  

4. **Tokenization**  
   - Tool: `nltk.word_tokenize`  
   - Why: Splits text into individual words (tokens) for further analysis  
   - What it does: `"Great app, love it"` → `["Great", "app", "love", "it"]`  

5. **Stopword Removal**  
   - Tool: `nltk.corpus.stopwords`  
   - Why: Removes very common words (like “the”, “is”, “an”) that don’t carry meaning  
   - What it does: `"I love the music"` → `["love", "music"]`  

6. **Lemmatization**  
   - Tool: `nltk.WordNetLemmatizer` (or spaCy)  
   - Why: Produces real, meaningful words by considering grammar and context, unlike stemming which is faster but less accurate  
   - What it does: Reduces words to their base or dictionary form (lemma), helping group similar words  

**Importance of these steps:**  

- Computers don’t understand raw text — they need a clean, structured representation.  
- These preprocessing steps ensure that models (VADER, TextBlob, DistilBERT, TF-IDF) can correctly analyze reviews for sentiment (positive/negative/neutral) and themes (keywords, frequent phrases).

---

## Sentiment Analysis

- No feature engineering was done; a pre-trained transformer model (DistilBERT) was used for end-to-end sentiment prediction.  

- **Tool Used:** Hugging Face Transformers – `pipeline("sentiment-analysis")` with DistilBERT fine-tuned on SST-2.  
- **Why:** Pre-trained on a large labeled dataset (SST-2), accurate for binary sentiment (positive/negative).  

**How it was applied:**  

- Predictions were run in batches on the `cleaned_review` column to get probabilities for POSITIVE and NEGATIVE.  
- Custom neutral rules:  
  - Confidence below 60% (low certainty)  
  - Probability difference between positive and negative < 20%  
  → Labeled as NEUTRAL  

**Outputs generated:**  

- `sent_pos_prob` – Positive probability  
- `sent_neg_prob` – Negative probability  
- `sent_confidence` – Model confidence  
- `Sentiment` – Final label (POSITIVE / NEGATIVE / NEUTRAL)  

---

## Thematic Analysis

- **Tool Used:** `TfidfVectorizer` from scikit-learn  
- **Why:** Converts text into numeric features representing the importance of each word or phrase, while remaining interpretable  

**What TF-IDF captures:**  

- Term Frequency (TF): how often a word appears in a review  
- Inverse Document Frequency (IDF): how rare the word is across all reviews, giving more weight to distinctive words  

**Feature engineering steps:**  

- Ignore extremely common words (`max_df=0.85`) and very rare words (`min_df=2`)  
- Include unigrams and bigrams  
- Remove stopwords  

**Outputs generated:**  

- Ranked list of top 20 keywords/phrases based on TF-IDF scores  
- Manual grouping of top terms into broader themes (e.g., App & Usability, Music & Songs, Ads & Premium)  
- Counts of how often each theme appeared in reviews  
- Visualizations of keyword importance and theme distribution  

---

## Comparison of Sentiment Analysis Tools

Three sentiment analysis tools—VADER, TextBlob, and DistilBERT—were compared against user ratings:  

- **DistilBERT:** Highest agreement (~72.7%), aligning best with user ratings  
- **VADER:** Moderate agreement (~63.6%), sometimes misclassifies neutral or slightly positive/negative reviews  
- **TextBlob:** Lowest agreement (~59.2%), often overly optimistic or neutral  

**Interpretation:**  

- Transformer-based models like DistilBERT capture context and subtle cues better, leading to higher alignment with actual ratings  
- Lexicon-based models like VADER and TextBlob are simpler and may over- or under-estimate sentiment, especially for nuanced or mixed reviews  

**Overall:** DistilBERT is the most reliable of the three for predicting sentiment that matches user ratings.
---

*© iCogLabs | August 20, 2025*
