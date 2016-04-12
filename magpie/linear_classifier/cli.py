import os

import click

from magpie.config import MODEL_PATH
from magpie.linear_classifier import api
from magpie.linear_classifier.config import ONTOLOGY_PATH, MODEL_PATH


@click.group()
def cli():
    """ A dummy function for grouping other commands """
    pass


@cli.command()
@click.argument('document')
@click.option(
    '--ontology',
    '-o',
    # prompt='Path to the ontology',
    default=ONTOLOGY_PATH,
    help='path to the ontology'
)
@click.option(
    '--model',
    '-m',
    # prompt='Path to the trained model',
    default=MODEL_PATH,
    help='path to the pickled model'
)
@click.option(
    '--recreate-ontology',
    default=False,
    help='whether to recreate the ontology'
)
@click.option(
    '--verbose/--quiet',
    '-v/-q',
    default=True,
    help='whether to display additional information e.g. computation time',
)
def extract(document, ontology, model, recreate_ontology, verbose):
    """ Extract keywords from a document """
    api.extract(
        os.path.abspath(document),
        ontology_path=os.path.abspath(ontology),
        model_path=os.path.abspath(model),
        recreate_ontology=recreate_ontology,
        verbose=verbose,
    )


@cli.command()
@click.argument('trainset_dir')
@click.option(
    '--ontology',
    '-o',
    # prompt='Path to the ontology',
    default=ONTOLOGY_PATH,
    help='path to the ontology'
)
@click.option(
    '--model',
    '-m',
    default=MODEL_PATH,
    help='path to the pickled model'
)
@click.option(
    '--recreate-ontology',
    is_flag=True,
    help='whether to recreate the ontology'
)
@click.option(
    '--verbose/--quiet',
    '-v/-q',
    default=True,
    help='whether to display additional information e.g. computation time',
)
def train(trainset_dir, ontology, model, recreate_ontology, verbose):
    """ Train a model on a given dataset """
    api.train(
        trainset_dir=os.path.abspath(trainset_dir),
        ontology_path=os.path.abspath(ontology),
        model_path=os.path.abspath(model),
        recreate_ontology=recreate_ontology,
        verbose=verbose,
    )
    click.echo(u"Training completed successfully!")


@cli.command()
@click.argument('testset_dir')
@click.option(
    '--ontology',
    '-o',
    # prompt='Path to the ontology',
    default=ONTOLOGY_PATH,
    help='path to the ontology'
)
@click.option(
    '--model',
    '-m',
    # prompt='Path to the trained model',
    default=MODEL_PATH,
    help='path to the pickled model'
)
@click.option(
    '--recreate-ontology',
    is_flag=True,
    help='whether to recreate the ontology'
)
@click.option(
    '--verbose/--quiet',
    '-v/-q',
    default=True,
    help='whether to display additional information e.g. computation time',
)
def test(testset_dir, ontology, model, recreate_ontology, verbose):
    """ Test a model on a given dataset """
    precision, recall, f1_score, accuracy = api.test(
        testset_path=os.path.abspath(testset_dir),
        ontology_path=os.path.abspath(ontology),
        model_path=os.path.abspath(model),
        recreate_ontology=recreate_ontology,
        verbose=verbose,
    )
    click.echo()
    click.echo(u"Precision: {0:.2f}%".format(precision * 100))
    click.echo(u"Recall: {0:.2f}%".format(recall * 100))
    click.echo(u"F1-score: {0:.2f}%".format(f1_score * 100))
    click.echo(u"Accuracy: {0:.2f}%".format(accuracy * 100))


@cli.command()
@click.argument('dataset_dir')
@click.option(
    '--recreate-ontology',
    is_flag=True,
    help='whether to recreate the ontology'
)
@click.option(
    '--verbose/--quiet',
    '-v/-q',
    default=True,
    help='whether to display additional information e.g. computation time',
)
def candidate_recall(dataset_dir, recreate_ontology, verbose):
    """ Calculate average recall for keyword candidate generation """
    recall = api.calculate_recall_for_kw_candidates(
        data_dir=os.path.abspath(dataset_dir),
        recreate_ontology=recreate_ontology,
        verbose=verbose,
    )
    click.echo(u"Candidate generation recall: {0:.2f}s".format(recall))


@cli.command()
@click.option(
    '--train_set',
    '-train',
    # prompt='Path to the training set',
    help='path to the training set'
)
@click.option(
    '--test_set',
    '-test',
    # prompt='Path to the test set',
    help='path to the test set'
)
@click.option(
    '--ontology',
    '-o',
    # prompt='Path to the ontology',
    default=ONTOLOGY_PATH,
    help='path to the ontology'
)
@click.option(
    '--model',
    '-m',
    default=MODEL_PATH,
    help='path to the pickled model'
)
@click.option(
    '--recreate-ontology',
    is_flag=True,
    help='whether to recreate the ontology'
)
@click.option(
    '--verbose/--quiet',
    '-v/-q',
    default=True,
    help='whether to display additional information e.g. computation time',
)
def train_and_test(
    train_set,
    test_set,
    ontology,
    model,
    recreate_ontology,
    verbose,
):
    """ An aggregate command for both training and testing. """
    api.train(
        trainset_dir=os.path.abspath(train_set),
        ontology_path=os.path.abspath(ontology),
        model_path=os.path.abspath(model),
        recreate_ontology=recreate_ontology,
        verbose=verbose,
    )
    click.echo(u"Training completed successfully!")

    precision, recall, f1_score, accuracy = api.test(
        testset_path=os.path.abspath(test_set),
        ontology_path=os.path.abspath(ontology),
        model_path=os.path.abspath(model),
        recreate_ontology=recreate_ontology,
        verbose=verbose,
    )
    click.echo()
    click.echo(u"Precision: {0:.2f}%".format(precision * 100))
    click.echo(u"Recall: {0:.2f}%".format(recall * 100))
    click.echo(u"F1-score: {0:.2f}%".format(f1_score * 100))
    click.echo(u"Accuracy: {0:.2f}%".format(accuracy * 100))
