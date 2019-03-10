from sklearn.feature_extraction.text import TfidfVectorizer
import normalizer as norm
from sklearn.externals import joblib
import pandas as pd

MODEL_FILENAME="tfidf_model.sav"
NORMALIZED_COLUMN = "text_normalized"

def train_TfIdf(df):
    corpus = list(df['text_normalized'])
    vectorizer = TfidfVectorizer()
    return vectorizer.fit(corpus)

def save_model(model):
    joblib.dump(model,MODEL_FILENAME)

def restore_model():
    return joblib.load(MODEL_FILENAME)

def apply_model(model, df):
    X_test = model.transform(df[NORMALIZED_COLUMN]).toarray()
    vocs = list(model.vocabulary_.keys())
    vocs = list(map(lambda x: "tfidf_"+x,vocs))
    X_test = pd.DataFrame(X_test, columns=vocs)
    return pd.concat([df,X_test], axis=1).drop([NORMALIZED_COLUMN,"text"], axis=1)

def run(file_train):
    print("Train file: %s"%(file_train))
    df = pd.read_csv(file_train)
    df = norm.pipeline(df)
    model = train_TfIdf(df)
    save_model(model)
    print("Model file %s"%(MODEL_FILENAME))
    print("TfIdf Model saved!")
    