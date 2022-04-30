# -*- coding: UTF-8 -*-

from Iris import *

from argparse import ArgumentParser

if __name__ == '__main__':
    #####

    # Parameters management
    args_parser = ArgumentParser(
        description="Iris-based biometric identification system"
    )
    args_parser.version = "1.0"

    group = args_parser.add_argument_group("data pre-processing")
    group.add_argument("-p", "--preprocessing", help="pre-processing", action="store_true")

    group = args_parser.add_argument_group("evaluation")
    group.add_argument("-e", "--evaluation", help="evaluation", action="store_true")
    group.add_argument("-s", "--sample", help="evaluation with sample of 10 people", action="store_true")
    group.add_argument("-f", "--folds", metavar="", help="number of folds", action="store")
    group.add_argument("-k", "--knn", metavar="", help="number of neighbors", action="store")

    group = args_parser.add_argument_group("export to csv files")
    group.add_argument("-c", "--csv", help="export to csv", action="store_true")
    group.add_argument("-tf", "--texturefeatures", help="export GLCM texture features", action="store_true")

    args = args_parser.parse_args()

    # Parameters validation & execution
    iris = Iris()

    if args.preprocessing:
        iris.pre_process_data()
    elif args.evaluation:
        if args.folds is None:
            if args.knn is None:
                iris.evaluation(args.sample)
            else:
                iris.evaluation(args.sample, n_knn=int(args.knn))
        else:
            if args.knn is None:
                iris.evaluation(args.sample, int(args.folds))
            else:
                iris.evaluation(args.sample, int(args.folds), int(args.knn))
    elif args.csv:
        if args.features:
            iris.export_glcm_pca_to_csv(args.sample)
        else:
            iris.create_glcm_features_csv_files(sample)
    else:
        args_parser.print_help()