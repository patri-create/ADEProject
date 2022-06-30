import csv
import torch
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoConfig, AutoModel
from utils.data import get_project_root, read_data, Token, Key
from flair.data import Sentence
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import (
    AccuracyCallback,
    CheckpointCallback,
    InferCallback,
    OptimizerCallback,
)
from catalyst.utils import prepare_cudnn, set_global_seed


class Extractor(torch.nn.Module):
    def __init__(
            self, pretrained_model_name: str, num_classes: int = None, dropout: float = 0.3
    ):
        super().__init__()

        config = AutoConfig.from_pretrained(
            pretrained_model_name, num_labels=num_classes
        )

        self.model = AutoModel.from_pretrained(pretrained_model_name, config=config)
        self.classifier = torch.nn.Linear(config.hidden_size, num_classes)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, features, attention_mask=None, head_mask=None):
        assert attention_mask is not None, "attention mask is none"
        bert_output = self.model(
            input_ids=features, attention_mask=attention_mask, head_mask=head_mask
        )
        seq_output = bert_output[0]
        pooled_output = seq_output.mean(axis=1)
        pooled_output = self.dropout(pooled_output)
        scores = self.classifier(pooled_output)

        return scores


project_root: Path = get_project_root()
with open(str(project_root / "config.yml")) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)


def train_extractor():
    train_file = params["extractor_data"]["iob2_train"]
    valid_file = params["extractor_data"]["iob2_valid"]
    test_file = params["extractor_data"]["iob2_test"]
    text = params["extractor_data"]["text_field"]
    label = params["extractor_data"]["label_field"]
    train_val_loaders, test_loaders = read_data(params,
                                                train_file, valid_file, test_file, text, label)

    model = Extractor(
        pretrained_model_name=params["model"]["model_name"],
        num_classes=params["model"]["extractor_classes"],
    )

    weights = [0.45, 0.45, 0.1]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(params["training"]["learn_rate"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    set_global_seed(params["general"]["seed"])
    prepare_cudnn(deterministic=True)

    runner = SupervisedRunner(input_key=("features", "attention_mask"))

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=train_val_loaders,
        callbacks=[
            AccuracyCallback(num_classes=int(params["model"]["extractor_classes"])),
            OptimizerCallback(accumulation_steps=int(params["training"]["accum_steps"])),
            CheckpointCallback(save_n_best=0)
        ],
        logdir=params["extractor_data"]["log_dir"],
        num_epochs=int(params["training"]["num_epochs"]),
        verbose=True,
    )
    use_extractor(test_loaders, model, runner)


def use_extractor(test_loaders=None, model=None, runner=None):
    if not test_loaders:
        test_file = params["extractor_data"]["iob2_test"]
        text = params["extractor_data"]["text_field"]
        test_loaders = read_data(params, test_file=test_file, text=text, all=False)

    if not model:
        model = Extractor(
            pretrained_model_name=params["model"]["model_name"],
            num_classes=params["model"]["extractor_classes"],
        )

    if not runner:
        runner = SupervisedRunner(input_key=("features", "attention_mask"))

    torch.cuda.empty_cache()
    runner.infer(
        model=model,
        loaders=test_loaders,
        callbacks=[
            CheckpointCallback(
                resume=f"{params['extractor_data']['log_dir']}/checkpoints/best.pth"
            ),
            InferCallback(),
        ],
        verbose=True,
    )

    predicted_scores = runner.callbacks[0].predictions["logits"]
    np.savetxt(X=predicted_scores, fname=params["extractor_data"]["test_pred_scores"])


def get_and_save_positives():
    df = pd.read_csv(params["classifier_data"]["dataset"])
    positive_dataset = df.loc[df[params["classifier_data"]["label_field"]] == 'hasADE']

    positive_dataset.to_csv(Path(params["extractor_data"]["positive_dataset"]), index=False)

    train, valid, test = np.split(positive_dataset.sample(frac=1),
                                  [int(0.7 * len(positive_dataset) + 1), int((0.15 + 0.7) * len(positive_dataset))])

    train.drop(columns=['idmeddra', 'meddra'], axis=1, inplace=True)
    valid.drop(columns=['idmeddra', 'meddra'], axis=1, inplace=True)
    test.drop(columns=['idmeddra', 'meddra'], axis=1, inplace=True)

    train.to_csv(Path(params["extractor_data"]["positive_train"]), index=False)
    valid.to_csv(Path(params["extractor_data"]["positive_valid"]), index=False)
    test.to_csv(Path(params["extractor_data"]["positive_test"]), index=False)


def get_master():
    master_df = pd.read_csv(Path(params["extractor_data"]["master"]))

    master = {}

    for idx, row in master_df.iterrows():
        if row['id'] in master:
            master[row['id']].append(Key(row['keywords'], row['start'], row['end']))
        else:
            master[row['id']] = [Key(row['keywords'], row['start'], row['end'])]

    return master


def write_IOB2_format():
    train_df = pd.read_csv(params["extractor_data"]["positive_train"])
    valid_df = pd.read_csv(params["extractor_data"]["positive_valid"])
    test_df = pd.read_csv(params["extractor_data"]["positive_test"])
    master = get_master()

    write_IOB2_format_aux(train_df, master, params["extractor_data"]["iob2_train"])
    write_IOB2_format_aux(valid_df, master, params["extractor_data"]["iob2_valid"])
    write_IOB2_format_aux(test_df, master, params["extractor_data"]["iob2_test"])


def write_IOB2_format_aux(input_df, master, output):
    headers = ['sentence', 'token', 'tag']

    token_list = []
    for idx, row in input_df.iterrows():
        if row['id'] in master:
            sent = Sentence(row['text'], use_tokenizer=True)
            keys = master[row['id']]
            prev = 'O'
            for token in sent:
                token_obj = Token(row['id'], token.text)
                for key in keys:
                    if token.start_pos >= key.start and token.end_pos <= key.end:
                        if token.start_pos == key.start or prev == 'O':
                            token_obj.tag = 'B-ADE'
                        else:
                            token_obj.tag = 'I-ADE'
                        break
                prev = token_obj.tag
                token_list.append(token_obj)

    with open(output, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(headers)
        for token in token_list:
            writer.writerow(token.get_fields())
