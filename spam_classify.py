
# https://www.kaggle.com/chandramoulinaidu/spam-classification-for-basic-nlp

import nltk
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


nltk.download('wordnet')
nltk.download('stopwords')

df = pd.read_csv('./data/spam_email_raw_text_for_NLP.csv')

tokenizer = nltk.RegexpTokenizer(r"\w+")

lemmatizer = WordNetLemmatizer()

stopwords = stopwords.words('english')

def message_to_token_list(s):
  tokens = tokenizer.tokenize(s)
  lowercased_tokens = [t.lower() for t in tokens]
  lemmatized_tokens = [lemmatizer.lemmatize(t) for t in lowercased_tokens]
  useful_tokens = [t for t in lemmatized_tokens if t not in stopwords]

  return useful_tokens

df = df.sample(frac=1, random_state=1)
df = df.reset_index(drop=True)

split_index = int(len(df) * 0.8)
train_df, test_df = df[:split_index], df[split_index:]

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

token_counter = {}

for message in train_df['MESSAGE']:
  message_as_token_lst = message_to_token_list(message)

  for token in message_as_token_lst:
    if token in token_counter:
      token_counter[token] += 1
    else:
      token_counter[token] = 1

def keep_token(proccessed_token, threshold):
  if proccessed_token not in token_counter:
    return False
  else:
    return token_counter[proccessed_token] > threshold

features = set()

for token in token_counter:
  if keep_token(token, 10000):
    features.add(token)

features = list(features)

token_to_index_mapping = {t:i for t, i in zip(features, range(len(features)))}

# "Bag of Words" (counts vector)

# ->  http  tr  size  3d  font  br  com  td   p   b
# ->    0    1    2    3   4    5    6    7   8   9
# ->   [0,   0,   0,   1,  2,   1,   2,   0,  0,  1]

#      [0.,  0.,  0.,   1., 2.,  1., 2.,  0., 0., 1.]

def message_to_count_vector(message):
  count_vector = np.zeros(len(features))

  processed_list_of_tokens = message_to_token_list(message)

  for token in processed_list_of_tokens:
    if token not in features:
      continue
    index = token_to_index_mapping[token]
    count_vector[index] += 1

  return count_vector

def df_to_X_y(dff):
  y = dff['CATEGORY'].to_numpy().astype(int)

  message_col = dff['MESSAGE']
  count_vectors = []

  for message in message_col:
    count_vector = message_to_count_vector(message)
    count_vectors.append(count_vector)

  X = np.array(count_vectors).astype(int)

  return X, y

X_train, y_train = df_to_X_y(train_df)

X_test, y_test = df_to_X_y(test_df)

scaler = MinMaxScaler().fit(X_train)

X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

lr = LogisticRegression().fit(X_train, y_train)
print(classification_report(y_test, lr.predict(X_test)))

# Compare logistic regression to random forest

rf = RandomForestClassifier().fit(X_train, y_train)
print(classification_report(y_test, rf.predict(X_test)))
