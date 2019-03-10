import click
import tfidf_model
import split_dataset as sd
import nb_model as nb

@click.command("split_dataset")
@click.option('--dataset', type=str, help="Dataset csv file")
@click.option('--test-size', type=float, help="Test size")
def split_dataset(dataset, test_size):
    sd.run(dataset, test_size)

@click.command("tfidf")
@click.option('--train', type=str, help="Train csv file")
def tfidf(train):
    tfidf_model.run(train)

@click.command("naive_bayes")
@click.option('--train', type=str, help="Train csv file")
@click.option('--test',  type=str, help="Test csv file")
def naive_bayes(train, test):
    nb.run(train, test)

