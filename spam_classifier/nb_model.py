import tfidf_model as TfIdf
import normalizer as norm
import pandas as pd
from sklearn.utils import resample
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.externals import joblib

MODEL_FILENAME="nb_model.sav"
TARGET_COLUMN = "class"

def __split_X_y(df):
    X = df.drop([TARGET_COLUMN], axis=1)
    y = df[TARGET_COLUMN]
    return (X, y)

def __upsampling(df, class_name):
    df_majority = df[df[TARGET_COLUMN]!=class_name]
    df_minority = df[df[TARGET_COLUMN]==class_name]
    n_upsample = df_majority.shape[0]
    df_minority_upsampled = resample(df_minority, n_samples=n_upsample)
    return pd.concat([df_minority_upsampled,df_majority],axis=0)

def __pipeline(df_train, df_test):
    model = TfIdf.restore_model()

    df_train = norm.pipeline(df_train)
    df_test = norm.pipeline(df_test)

    df_train = TfIdf.apply_model(model,df_train)
    df_test = TfIdf.apply_model(model,df_test)

    df_train_upsampling = __upsampling(df_train, "spam")

    X_train, y_train = __split_X_y(df_train_upsampling)
    X_test, y_test = __split_X_y(df_test)

    return (X_train, y_train, X_test, y_test)

def create_model(df_train, df_test):
    X_train, y_train, X_test, y_test = __pipeline(df_train, df_test)
    model = BernoulliNB()
    model = model.fit(X_train,y_train)
    df_test["predict"] = model.predict(X_test)
    print(classification_report(y_test,df_test["predict"]))
    print("Acc: %s"%(accuracy_score(y_test,df_test["predict"])))
    return model



def run(file_train, file_test):
    df_train = pd.read_csv(file_train)
    df_test = pd.read_csv(file_test)

    model = create_model(df_train, df_test)
    joblib.dump(model,MODEL_FILENAME)
    print("Model filename %s"%(MODEL_FILENAME))
    print("NB Model Saved!")



