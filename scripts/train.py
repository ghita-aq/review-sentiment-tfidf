# Training pipeline: load, split, TF-IDF, train, evaluate

"""
 End-to-end training script that:
  - Loads the dataset from `data/reviews.csv`
  - Preprocesses text
  - Splits into train/test (80/20, stratified, random_state=42)
  - Fits TF-IDF vectorizer on training data
  - Trains three classifiers (Naive Bayes, Random Forest, SVM)
  - Evaluates each and prints comparison metrics
  - Saves the best model, vectorizer, and label encoder to `models/`
"""