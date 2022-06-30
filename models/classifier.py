import torch
from transformers import AutoConfig, AutoModel
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import (
    AccuracyCallback,
    CheckpointCallback,
    InferCallback,
    OptimizerCallback,
)

from catalyst.utils import prepare_cudnn, set_global_seed

from utils.data import read_data, get_project_root


class Classifier(torch.nn.Module):
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


def train_classifier():
    train_file = params["classifier_data"]["train"]
    valid_file = params["classifier_data"]["valid"]
    test_file = params["classifier_data"]["test"]
    text = params["classifier_data"]["text_field"]
    label = params["classifier_data"]["label_field"]
    train_val_loaders, test_loaders = read_data(params,
                                                train_file, valid_file, test_file, text, label)

    model = Classifier(
        pretrained_model_name=params["model"]["model_name"],
        num_classes=params["model"]["classifier_classes"],
    )

    criterion = torch.nn.CrossEntropyLoss()
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
            AccuracyCallback(num_classes=int(params["model"]["classifier_classes"])),
            OptimizerCallback(accumulation_steps=int(params["training"]["accum_steps"])),
            CheckpointCallback(save_n_best=0)
        ],
        logdir=params["classifier_data"]["log_dir"],
        num_epochs=int(params["training"]["num_epochs"]),
        verbose=True,
    )
    use_classifier(test_loaders, model, runner)


def use_classifier(test_loaders=None, model=None, runner=None):
    if not test_loaders:
        test_file = params["classifier_data"]["test"]
        text = params["classifier_data"]["text_field"]
        test_loaders = read_data(params, test_file=test_file, text=text, all=False)

    if not model:
        model = Classifier(
            pretrained_model_name=params["model"]["model_name"],
            num_classes=params["model"]["classifier_classes"],
        )

    if not runner:
        runner = SupervisedRunner(input_key=("features", "attention_mask"))

    torch.cuda.empty_cache()
    runner.infer(
        model=model,
        loaders=test_loaders,
        callbacks=[
            CheckpointCallback(
                resume=f"{params['classifier_data']['log_dir']}/checkpoints/best.pth"
            ),
            InferCallback(),
        ],
        verbose=True,
    )

    predicted_scores = runner.callbacks[0].predictions["logits"]
    np.savetxt(X=predicted_scores, fname=params["classifier_data"]["test_pred_scores"])


def generate_and_save_train_valid_test():
    df = pd.read_csv(params["classifier_data"]["dataset"])

    train, valid, test = np.split(df.sample(frac=1),
                                  [int(0.7 * len(df) + 1), int((0.15 + 0.7) * len(df))])

    train.drop(columns=['start', 'end', 'keywords', 'idmeddra', 'meddra'], axis=1, inplace=True)
    valid.drop(columns=['start', 'end', 'keywords', 'idmeddra', 'meddra'], axis=1, inplace=True)
    test.drop(columns=['start', 'end', 'keywords', 'idmeddra', 'meddra'], axis=1, inplace=True)

    train.to_csv(Path(params["classifier_data"]["train"]), index=False)
    valid.to_csv(Path(params["classifier_data"]["valid"]), index=False)
    test.to_csv(Path(params["classifier_data"]["test"]), index=False)
