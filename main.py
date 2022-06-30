from models.classifier import train_classifier, use_classifier, generate_and_save_train_valid_test
from models.extractor import train_extractor, use_extractor, get_and_save_positives, write_IOB2_format
from models.normalizer import train_normalizer, use_normalizer, get_train_valid_and_test, use_normalizer_similarity
from utils.evaluation import evaluate
from utils.data import Statistics
import argparse
import sys
import logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('op', type=str, choices=['classify', 'extract', 'normalize', 'similarity'],
                        help='operation to perform')
    parser.add_argument('mode', type=str, choices=['dataset', 'train', 'use', 'evaluate', 'analysis'],
                        help='generate dataset, training mode, use the trained model, evaluation mode or making a '
                             'data analysis.')
    # Parse arguments
    try:
        args = parser.parse_args()
        logging.info("Input Arguments : %s", args)
    except:
        parser.print_help()
        sys.exit(0)

    if args.op == "classify":
        if args.mode == 'dataset':
            generate_and_save_train_valid_test()
        if args.mode == 'train':
            train_classifier()
        if args.mode == 'use':
            use_classifier()
        if args.mode == 'evaluate':
            evaluate('classifier')
        if args.mode == 'analysis':
            Statistics('classifier').analysis()
    elif args.op == 'extract':
        if args.mode == 'dataset':
            get_and_save_positives()
            write_IOB2_format()
        if args.mode == 'train':
            train_extractor()
        if args.mode == 'use':
            use_extractor()
        if args.mode == 'evaluate':
            evaluate('extractor')
        if args.mode == 'analysis':
            Statistics('extractor').analysis()
    elif args.op == 'normalize':
        if args.mode == 'dataset':
            get_train_valid_and_test()
        if args.mode == 'train':
            train_normalizer()
        if args.mode == 'evaluate':
            evaluate('normalizer')
        if args.mode == 'use':
            use_normalizer()
        if args.mode == 'analysis':
            Statistics('normalizer').analysis()
    elif args.op == 'normalize':
        if args.mode == 'dataset':
            get_train_valid_and_test()
        if args.mode == 'evaluate':
            evaluate('normalizer')
        if args.mode == 'use':
            use_normalizer_similarity()
        if args.mode == 'analysis':
            Statistics('similarity').analysis()


if __name__ == '__main__':
    main()
