import spacy
import pandas as pd

nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])


def normalize_text(text):
    doc = nlp(text)
    result = [token.norm_ for token in doc if ((token.is_stop is False) and (token.is_punct is False)) ]
    return " ".join(result)

def df_normalize_text_column(df):
    df['text_normalized'] = df['text'].apply(normalize_text)
    return df

def pipeline(df):
    return df_normalize_text_column(df)
