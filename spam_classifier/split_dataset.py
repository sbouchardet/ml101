import pandas as pd
from sklearn.model_selection import train_test_split

def run(dataset_file, test_size):
    df_spam = pd.read_csv(dataset_file, encoding='latin-1')[["v1","v2"]].rename(columns={'v1':'class', 'v2':'text'}) 
    train, test = train_test_split(df_spam, test_size=test_size)
    train.to_csv(dataset_file.split(".")[0]+"_train.csv",index=False)
    test.to_csv(dataset_file.split(".")[0]+"_test.csv",index=False)