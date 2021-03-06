import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import re


def predict(email):
    if re.search("urgent", email["text"], re.IGNORECASE):
        return "spam"
    else:
        return "ham"

def run(file_test):
    dataframe = pd.read_csv(file_test)
    dataframe["predict"] = dataframe.apply(predict, axis=1)
    print(classification_report(dataframe["class"],dataframe["predict"]))
    print("Acc: %s"%(accuracy_score(dataframe["class"],dataframe["predict"])))
