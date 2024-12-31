"""Functions to launch training from the command line"""

import sys
import argparse
from typing import Optional
import logging

from src.libs.preprocessing import DataLoader

from src.model.experiments import init_pipeline_from_config, load_pipeline_from_config


def get_parser(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """
    Create parser to run training from terminal.

    Args:
        parser (Optional[argparse.ArgumentParser], optional): Parser. Defaults to None.

    Returns:
        argparse.ArgumentParser: Parser with the new arguments.
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Train a model")
    # Experiment ID
    parser.add_argument(
        "-e", "--exp", nargs="+", type=int, required=True, help="Experiment id"
    )
    # Load a pretrained pipeline
    parser.add_argument("--load", action="store_true", help="Load pretrained pipeline")
    # Local flag
    parser.add_argument(
        "--local_data", action="store_true", help="Load data from local filesystem"
    )
    # Learning flag
    parser.add_argument(
        "--learning", action="store_true", help="Whether to launch learning or not"
    )
    # Testing flag
    parser.add_argument(
        "--testing", action="store_true", help="Whether to launch testing or not"
    )
    # Full flag (learning and testing)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Whether to launch learning and testing or not",
    )
    return parser


def get_data(train_small: bool = False):
    """Load data"""
    # - train data
    path_train_x = (
        "data/input/denoising/train_small"
        if train_small
        else "data/input/denoising/train"
    )
    path_train_y = (
        "data/input/voice_origin/train_small"
        if train_small
        else "data/input/voice_origin/train"
    )
    data_loader = DataLoader(path_x=path_train_x, path_y=path_train_y)
    data_train = data_loader.get_harmonized_data(downsample=train_small)
    del data_loader
    # - test data
    path_test_x = "data/input/denoising/test"
    path_test_y = "data/input/voice_origin/test"
    data_loader = DataLoader(path_x=path_test_x, path_y=path_test_y)
    data_test = data_loader.get_harmonized_data()
    del data_loader
    print("Data loaded!")
    return data_train, data_test


def train_main(argv):
    """
    Launch training from terminal.

    Args:
        argv (_type_): Parser arguments.
    """
    parser = get_parser()
    args = parser.parse_args(argv)
    print("Args Local Data", args.local_data)
    data_train, data_test = get_data()
    for exp in args.exp:
        if args.load:
            pipeline = load_pipeline_from_config(exp)
        else:
            pipeline = init_pipeline_from_config(exp)
        logging.info("Training experiment %s", exp)
        if args.full:
            pipeline.full_pipeline(data_train, data_test)
        elif args.learning:
            pipeline.learning_pipeline(data_train, data_test)
        elif args.testing:
            pipeline.testing_pipeline(data_train, data_test)


if __name__ == "__main__":
    train_main(sys.argv[1:])
