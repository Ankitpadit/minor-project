# Sentiment Analysis of Twitter Posts (ML Version)

## Objective
Analyze sentiment (positive, negative, neutral) of Tweets using a machine learning model.

## Features
- Fetch tweets by keyword/hashtag using Twitter API
- Clean and preprocess tweet text
- Train a Logistic Regression sentiment classifier on labelled data
- Predict sentiment of live tweets
- Evaluate model accuracy and F1 score

## Files
- `main.py`: Main script (training + prediction)
- `sample_data.csv`: Sample labelled data for training
- `requirements.txt`: Dependencies

## How to Run

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
2. **Get Twitter API keys** and add them in `main.py`.
3. **Train or use model:**
   - First run will train model on `sample_data.csv`.
   - On next runs, model will be loaded from disk.
4. **Analyze tweets:**
   ```
   python main.py
   ```
   - Enter `1` for tweet analysis, `2` for retraining model.

## Note
- For better accuracy, expand `sample_data.csv` with more labelled data.
- You can use public datasets like [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) for more training data.

## Output
- Sentiment predictions for fetched tweets
- Summary count of predicted sentiments
- Model accuracy and F1 score (on train/test split)

## Example
```
Enter 1 to analyze tweets, 2 to train model: 1
Enter keyword or hashtag: apple
positive: Apple is amazing!
negative: I hate apple's new update.
neutral: Apple releases new product.
...
positive    7
negative    2
neutral     1
dtype: int64
```
