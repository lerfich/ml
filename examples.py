from scipy.sparse.sputils import validateaxis
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

#example 1: get features 
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print("Features:")
print(vectorizer.get_feature_names_out())
pairwise_sim = X*X.T
print("\n Similarity matrix:")
print(pairwise_sim.toarray())

#example 2: weight values display
sentences = [
    "Three years later, the coffin was still full of Jello.",
    "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
    "The person box was packed with jelly many dozens of months later.",
    "He found a leprechaun in his walnut shell."
]

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(sentences)
feature_names = vectorizer.get_feature_names_out()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)

#  df["max"] = df.max(axis=1)
#  df["sum"] = df.sum(axis=1)

print("\n Tf-idf values:")
print(df)
pairwise_sim = vectors*vectors.T
print("\n Similarity matrix:")
print(pairwise_sim.toarray())