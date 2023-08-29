import argparse

from spine.models import tuning

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--directory",
        type=str,
        default="../data/processed/",
        help="The path to the directory containing the data files.",
    )
    argparser.add_argument(
        "--model_type",
        type=str,
        default="BaselineModel",
        help="The type of model to use.",
    )

    args = argparser.parse_args()

    tuner = tuning.Tuner(
        directory=args.directory,
        input_size=21,
        output_size=12,
        model_type=args.model_type,
        epochs=10,
        n_trials=10,
    )
    tuner.tune()
