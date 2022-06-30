import logging
from pathlib import Path
import yaml
from typing import List, Mapping, Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from catalyst.utils import set_global_seed
import matplotlib.pyplot as plt


class DatasetWrapper(Dataset):

    def __init__(
            self,
            texts: List[str],
            labels: List[str] = None,
            label_dict: Mapping[str, int] = None,
            max_seq_length: int = 512,
            model_name: str = "distilbert-base-uncased",
    ):

        self.texts = texts
        self.labels = labels
        self.label_dict = label_dict
        self.max_seq_length = max_seq_length

        if self.label_dict is None and labels is not None:
            self.label_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.getLogger("transformers.tokenization_utils").setLevel(logging.FATAL)

        self.sep_vid = self.tokenizer.vocab["[SEP]"]
        self.cls_vid = self.tokenizer.vocab["[CLS]"]
        self.pad_vid = self.tokenizer.vocab["[PAD]"]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index) -> Mapping[str, torch.Tensor]:
        x = self.texts[index]

        output_dict = self.tokenizer.encode_plus(
            x,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
            truncation=True,
            return_attention_mask=True,
        )

        output_dict["features"] = output_dict["input_ids"].squeeze(0)
        del output_dict["input_ids"]

        if self.labels is not None:
            y = self.labels[index]
            y_encoded = torch.Tensor([self.label_dict.get(y, -1)]).long().squeeze(0)
            output_dict["targets"] = y_encoded

        return output_dict


def read_data(params: dict, train_file=None, valid_file=None, test_file=None, text=None, label=None, all=True) -> Tuple[
    dict, dict]:
    if all:
        train_df = pd.read_csv(train_file)
        valid_df = pd.read_csv(valid_file)
    test_df = pd.read_csv(test_file)

    if all:
        train_dataset = DatasetWrapper(
            texts=train_df[text].values.tolist(),
            labels=train_df[label].values,
            max_seq_length=params["model"]["max_seq_length"],
            model_name=params["model"]["model_name"],
        )

        valid_dataset = DatasetWrapper(
            texts=valid_df[text].values.tolist(),
            labels=valid_df[label].values,
            max_seq_length=params["model"]["max_seq_length"],
            model_name=params["model"]["model_name"],
        )

    test_dataset = DatasetWrapper(
        texts=test_df[text].values.tolist(),
        # labels=test_df[label].values,
        max_seq_length=params["model"]["max_seq_length"],
        model_name=params["model"]["model_name"],
    )

    set_global_seed(params["general"]["seed"])

    if all:
        train_val_loaders = {
            "train": DataLoader(
                dataset=train_dataset,
                batch_size=params["training"]["batch_size"],
                shuffle=True,
            ),
            "valid": DataLoader(
                dataset=valid_dataset,
                batch_size=params["training"]["batch_size"],
                shuffle=False,
            ),
        }

    test_loaders = {
        "test": DataLoader(
            dataset=test_dataset,
            batch_size=params["training"]["batch_size"],
            shuffle=False,
        )
    }

    if all:
        return train_val_loaders, test_loaders
    else:
        return test_loaders


class Text:
    def __init__(self, tid, text):
        self.tid = tid
        self.text = text


def get_project_root() -> Path:
    return Path(__file__).parent.parent


class Token:
    def __init__(self, sentence: str, token: str, tag: str = 'O'):
        self.sentence = sentence
        self.token = token
        self.tag = tag

    def get_fields(self):
        return [self.sentence, self.token, self.tag]


class Key:
    def __init__(self, keywords: str, start: int, end: int):
        self.keywords = keywords
        self.start = start
        self.end = end

    def get_fields(self):
        return [self.keywords, self.start, self.end]


project_root: Path = get_project_root()
with open(str(project_root / "config.yml")) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)


class ClassifierAnalysis:
    def __init__(self):
        self.total = 0
        self.positives = 0
        self.negatives = 0
        self.train = 0
        self.valid = 0
        self.test = 0
        self.train_positives = 0
        self.train_negatives = 0
        self.valid_positives = 0
        self.valid_negatives = 0
        self.test_positives = 0
        self.test_negatives = 0

    def analysis(self):
        dataset = pd.read_csv(params['classifier_data']['dataset'])
        train = pd.read_csv(params['classifier_data']['train'])
        valid = pd.read_csv(params['classifier_data']['valid'])
        test = pd.read_csv(params['classifier_data']['test'])

        positive_dataset = dataset.loc[dataset[params["classifier_data"]["label_field"]] == 'hasADE']
        positive_train = train.loc[train[params["classifier_data"]["label_field"]] == 'hasADE']
        positive_valid = valid.loc[valid[params["classifier_data"]["label_field"]] == 'hasADE']
        positive_test = test.loc[test[params["classifier_data"]["label_field"]] == 'hasADE']

        self.total = len(dataset)
        self.train = len(train)
        self.valid = len(valid)
        self.test = len(test)
        self.positives = len(positive_dataset)
        self.negatives = self.total - self.positives
        self.train_positives = len(positive_train)
        self.train_negatives = self.train - self.train_positives
        self.valid_positives = len(positive_valid)
        self.valid_negatives = self.valid - self.valid_positives
        self.test_positives = len(positive_test)
        self.test_negatives = self.test - self.test_positives

        self.show_analysis()

    def show_analysis(self):
        print(self.get_info())
        show_simple_analysis(self.train, self.valid, self.test)

        labels = ['ADE', 'NoADE']
        sizes = [self.positives, self.negatives]
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%',
               shadow=False, startangle=90)

        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        ax.axis('equal')
        plt.tight_layout()
        plt.show()

    def get_info(self):
        return '***** CLASSIFIER DATA *****\n' \
               f'total:\t{self.total}\n' \
               f'train:\t{self.train}\n' \
               f'valid:\t{self.valid}\n' \
               f'test:\t{self.test}\n\n' \
               '\tPOS\tNEG\n' \
               f'total:\t{self.positives}\t{self.negatives}\n' \
               f'train:\t{self.train_positives}\t{self.train_negatives}\n' \
               f'valid:\t{self.valid_positives}\t{self.valid_negatives}\n' \
               f'test:\t{self.test_positives}\t{self.test_negatives}\n\n'


class ExtractorAnalysis:
    def __init__(self):
        self.total = 0
        self.train = 0
        self.valid = 0
        self.test = 0
        self.total_o_tokens = 0
        self.total_ade_tokens = 0
        self.train_o_tokens = 0
        self.train_ade_tokens = 0
        self.valid_o_tokens = 0
        self.valid_ade_tokens = 0
        self.test_o_tokens = 0
        self.test_ade_tokens = 0

    def analysis(self):
        master = pd.read_csv(params['extractor_data']['master'])
        train = pd.read_csv(params['extractor_data']['positive_train'])
        valid = pd.read_csv(params['extractor_data']['positive_valid'])
        test = pd.read_csv(params['extractor_data']['positive_test'])
        iob2_train = pd.read_csv(params['extractor_data']['iob2_train'])
        iob2_valid = pd.read_csv(params['extractor_data']['iob2_valid'])
        iob2_test = pd.read_csv(params['extractor_data']['iob2_test'])
        iob2_train_o_tokens = iob2_train.loc[iob2_train[params["extractor_data"]["label_field"]] == 'O']
        iob2_valid_o_tokens = iob2_valid.loc[iob2_valid[params["extractor_data"]["label_field"]] == 'O']
        iob2_test_o_tokens = iob2_test.loc[iob2_test[params["extractor_data"]["label_field"]] == 'O']

        self.total = len(master)
        self.train = len(train)
        self.valid = len(valid)
        self.test = len(test)
        self.total_o_tokens = len(iob2_train_o_tokens) + len(iob2_valid_o_tokens) + len(iob2_test_o_tokens)
        self.total_ade_tokens = (len(iob2_train) + len(iob2_valid) + len(iob2_test)) - self.total_o_tokens
        self.train_o_tokens = len(iob2_train_o_tokens)
        self.train_ade_tokens = len(iob2_train) - self.train_o_tokens
        self.valid_o_tokens = len(iob2_valid_o_tokens)
        self.valid_ade_tokens = len(iob2_valid) - self.valid_o_tokens
        self.test_o_tokens = len(iob2_test_o_tokens)
        self.test_ade_tokens = len(iob2_test) - self.test_o_tokens

        self.show_analysis()

    def show_analysis(self):
        print(self.get_info())
        show_simple_analysis(self.train, self.valid, self.test)
        labels = ['O tokens', 'ADE tokens']
        sizes = [self.total_o_tokens, self.total_ade_tokens]
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                shadow=False, startangle=90)

        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        ax1.axis('equal')
        plt.tight_layout()
        plt.show()

    def get_info(self):
        return '***** EXTRACTOR DATA *****\n' \
               f'total:\t{self.total}\n' \
               f'train:\t{self.train}\n' \
               f'valid:\t{self.valid}\n' \
               f'test:\t{self.test}\n\n' \
               'tokens\tO\tADE\n' \
               f'total:\t{self.total_o_tokens}\t{self.total_ade_tokens}\n' \
               f'train:\t{self.train_o_tokens}\t{self.train_ade_tokens}\n' \
               f'valid:\t{self.valid_o_tokens}\t{self.valid_ade_tokens}\n' \
               f'test:\t{self.test_o_tokens}\t{self.test_ade_tokens}\n'


class NormalizerAnalysis:
    def __init__(self):
        self.total = 0
        self.train = 0
        self.valid = 0
        self.test = 0
        self.total_different_labels = 0
        self.train_different_labels = 0
        self.valid_different_labels = 0
        self.test_different_labels = 0

    def analysis(self):
        master = pd.read_csv(params['normalizer_data']['master'])
        train = pd.read_csv(params['normalizer_data']['train'])
        valid = pd.read_csv(params['normalizer_data']['valid'])
        test = pd.read_csv(params['normalizer_data']['test'])

        self.total = len(master)
        self.train = len(train)
        self.valid = len(valid)
        self.test = len(test)
        self.total_different_labels = len(master[params['normalizer_data']['label_field']].unique())
        self.train_different_labels = len(train[params['normalizer_data']['label_field']].unique())
        self.valid_different_labels = len(valid[params['normalizer_data']['label_field']].unique())
        self.test_different_labels = len(test[params['normalizer_data']['label_field']].unique())

        self.show_analysis()

    def show_analysis(self):
        print(self.get_info())
        show_simple_analysis(self.train, self.valid, self.test)
        labels = ['Total data', 'Different labels in data']
        sizes = [self.total, self.total_different_labels]
        plt.rcdefaults()
        fig, ax = plt.subplots()
        ax.bar(
            x=[1, 2],
            height=sizes,
            tick_label=labels
        )
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#EEEEEE')
        ax.xaxis.grid(False)
        plt.show()

    def get_info(self):
        return '***** NORMALIZER DATA *****\n' \
               f'total:\t{self.total}\n' \
               f'train:\t{self.train}\n' \
               f'valid:\t{self.valid}\n' \
               f'test:\t{self.test}\n\n' \
               '\tDIFFERENT LABELS\n' \
               f'total:\t{self.total_different_labels}\n' \
               f'train:\t{self.train_different_labels}\n' \
               f'valid:\t{self.valid_different_labels}\n' \
               f'test:\t{self.test_different_labels}\n\n'


def show_simple_analysis(train, valid, test):
    labels = ['train', 'valid', 'test']
    sizes = [train, valid, test]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=False, startangle=90)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    ax1.axis('equal')
    plt.tight_layout()
    plt.show()


class Statistics:
    def __init__(self, model: str = ''):
        self.model = model
        self.classifier = ClassifierAnalysis()
        self.extractor = ExtractorAnalysis()
        self.normalizer = NormalizerAnalysis()

    def analysis(self):
        if self.model == 'classifier':
            self.classifier.analysis()
        elif self.model == 'extractor':
            self.extractor.analysis()
        elif self.model == 'normalizer':
            self.normalizer.analysis()
        else:
            print('Error in Statistics, please, choose classifier, extractor or normalizer')
