# Spam classifier

This project is an example used to present the lecture [Machine Learning 101](https://github.com/sbouchardet/ml101/blob/master/MachineLearning_101.pdf).

The **Spam classifier** example creates a model to classify emails based on its texts.

## Setup

### Create virtualenv

```bash
virtualenv -p python3 myenv
source myenv/bin/activate
```

### Install dependencies

```bash
make install
```

## Run classification steps

### Split dataset

```bash
split_dataset --dataset spam.csv --test-size 0.2
```
This step must create two files: `spam_test.csv` and `spam_train.csv`.

### Create Tf-Idf model

```bash
tfidf --train spam_train.csv
```

### Create and eval the model

```bash
naive_bayes --train spam_train.csv --test spam_test.csv
```