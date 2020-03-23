import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# read json into a dataframe
df_idf=pd.read_csv("Tweets.csv")

# print schema
print("Schema:\n\n",df_idf.dtypes)
print("Number of questions,columns=",df_idf.shape)


def pre_process(text):
    # lowercase
    text = text.lower()

    # remove tags
    text = re.sub("</?.*?>", " <> ", text)

    # remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)

    text = re.sub(r'https?:\/\/\S+', '', text)  # remove the links

    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # remove @mentions

    text = re.sub(r'#', '', text)  # remove # symbol

    return text


df_idf['text'] = df_idf['text'].apply(lambda x: pre_process(x))

# show the first 'text'
print(df_idf['text'])


# get the text column
docs = df_idf['text'].tolist()

# create a vocabulary of words,
# ignore words that appear in 85% of documents,
# eliminate stop words
cv = CountVectorizer(max_df=0.85, stop_words='english')
word_count_vector = cv.fit_transform(docs)

print(word_count_vector.shape)
print(list(cv.vocabulary_.keys())[:10])



tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)

# read test docs into a dataframe and concatenate title and body
df_test = pd.read_csv("Tweets-test.csv")
df_test['text'] = df_test['text'].apply(lambda x: pre_process(x))

# get test docs into a list
docs_test = df_test['text'].tolist()

# you only needs to do this once, this is a mapping of index to
feature_names = cv.get_feature_names()

# get the document that we want to extract keywords from
doc = docs_test[15]


# generate tf-idf for the given document
tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))
print(tf_idf_vector)



def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

# sort the tf-idf vectors by descending order of scores
sorted_items = sort_coo(tf_idf_vector.tocoo())




def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    # results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results

# extract only the top n; n here is 10
keywords = extract_topn_from_vector(feature_names, sorted_items, 10)



# print the results
print("\n=====Doc=====")
print(doc)
print("\n===Keywords===")
for k in keywords:
    print(k, keywords[k])





