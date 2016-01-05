import os

import click

from magpie import api
from magpie.config import HEP_ONTOLOGY, MODEL_PATH


@click.group()
def cli():
    pass


@cli.command()
@click.argument('document')
@click.option(
    '--ontology',
    '-o',
    # prompt='Path to the ontology',
    default=HEP_ONTOLOGY,
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
    '--show-answers',
    default=False,
    help='whether to load and show ground truth answers'
)
@click.option(
    '--recreate-ontology',
    default=False,
    help='whether to recreate the ontology'
)
def extract(document, ontology, model, show_answers, recreate):
    """ Extract keywords from a document """
    api.extract(
        os.path.abspath(document),
        ontology_path=os.path.abspath(ontology),
        model_path=os.path.abspath(model),
        show_answers=show_answers,
        recreate_ontology=recreate
    )


@cli.command()
@click.argument('trainset_dir')
@click.option(
    '--ontology',
    '-o',
    # prompt='Path to the ontology',
    default=HEP_ONTOLOGY,
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
    default=False,
    help='whether to recreate the ontology'
)
def train(trainset_dir, ontology, model, recreate_ontology):
    """ Train a model on a given dataset """
    api.train(
        trainset_dir=os.path.abspath(trainset_dir),
        ontology_path=os.path.abspath(ontology),
        model_path=os.path.abspath(model),
        recreate_ontology=recreate_ontology
    )


@cli.command()
@click.argument('testset_dir')
@click.option(
    '--ontology',
    '-o',
    # prompt='Path to the ontology',
    default=HEP_ONTOLOGY,
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
def test(testset_dir, ontology, model, recreate_ontology):
    """ Test a model on a given dataset """
    api.test(
        testset_path=os.path.abspath(testset_dir),
        ontology_path=os.path.abspath(ontology),
        model_path=os.path.abspath(model),
        recreate_ontology=recreate_ontology
    )


@cli.command()
@click.argument('dataset_dir')
@click.option(
    '--recreate-ontology',
    default=False,
    help='whether to recreate the ontology'
)
def candidate_recall(dataset_dir, recreate_ontology):
    """ Calculate average recall for keyword candidate generation """
    api.calculate_recall_for_kw_candidates(
        data_dir=os.path.abspath(dataset_dir),
        recreate_ontology=recreate_ontology
    )
