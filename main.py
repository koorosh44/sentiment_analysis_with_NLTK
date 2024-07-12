import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

a = "This is a good movie"
sid.polarity_scores(a)

b = "This was the best, most awsome movie EVER MADE!!!"
sid.polarity_scores(b)

c = "This was the WORST movie that has ever disgraced the screen."
sid.polarity_scores(c)

# Use VADER to analyze Amazon Reviews

import pandas as pd

df = pd.read_csv('TextFiles/amazonreviews.tsv',sep='\t')

df.head()

df['label'].value_counts()

# Clean the data

df.dropna(inplace=True)

blanks = []
for i,lb,rv in df.itertuples():
    #(index,label,review)
    if type(rv) == str:
        if rv.isspace():
            blanks.append(i)

blanks

df.iloc[0]['review']

sid.polarity_scores(df.iloc[0]['review'])

# Adding Scores and Labels to the DataFrame
df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review))
df.head()
df['compound'] = df['scores'].apply(lambda d:d['compound'])
df.head()

df['compound_score'] = df['compound'].apply(lambda score: 'pos' if score >=0 else 'neg')
df.head()

# Report on Accuracy
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
accuracy_score(df['label'],df['compound_score'])
print(classification_report(df['label'],df['compound_score']))
print(confusion_matrix(df['label'],df['compound_score']))
