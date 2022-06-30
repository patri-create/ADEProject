import torch
from transformers import AutoConfig, AutoModel
from pathlib import Path
import numpy as np
import yaml
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import (
    AccuracyCallback,
    CheckpointCallback,
    InferCallback,
    OptimizerCallback,
)

from catalyst.utils import prepare_cudnn, set_global_seed
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import trange

from utils.data import read_data, get_project_root


class Normalizer(torch.nn.Module):
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


def train_normalizer():
    train_file = params["normalizer_data"]["train"]
    valid_file = params["normalizer_data"]["valid"]
    test_file = params["normalizer_data"]["test"]
    text = params["normalizer_data"]["text_field"]
    label = params["normalizer_data"]["label_field"]
    train_val_loaders, test_loaders = read_data(params,
                                                train_file, valid_file, test_file, text, label)

    model = Normalizer(
        pretrained_model_name=params["model"]["model_name"],
        num_classes=params["model"]["normalizer_classes"],
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
            AccuracyCallback(num_classes=int(params["model"]["normalizer_classes"])),
            OptimizerCallback(accumulation_steps=int(params["training"]["accum_steps"])),
            CheckpointCallback(save_n_best=0)
        ],
        logdir=params["normalizer_data"]["log_dir"],
        num_epochs=int(params["training"]["num_epochs"]),
        verbose=True,
    )
    use_normalizer(test_loaders, model, runner)


def use_normalizer(test_loaders=None, model=None, runner=None):
    if not test_loaders:
        test_file = params["normalizer_data"]["test"]
        text = params["normalizer_data"]["text_field"]
        test_loaders = read_data(params, test_file=test_file, text=text, all=False)

    if not model:
        model = Normalizer(
            pretrained_model_name=params["model"]["model_name"],
            num_classes=params["model"]["normalizer_classes"],
        )

    if not runner:
        runner = SupervisedRunner(input_key=("features", "attention_mask"))

    torch.cuda.empty_cache()
    runner.infer(
        model=model,
        loaders=test_loaders,
        callbacks=[
            CheckpointCallback(
                resume=f"{params['normalizer_data']['log_dir']}/checkpoints/best.pth"
            ),
            InferCallback(),
        ],
        verbose=True,
    )

    predicted_scores = runner.callbacks[0].predictions["logits"]
    np.savetxt(X=predicted_scores, fname=params["normalizer_data"]["test_pred_scores"])


def use_normalizer_similarity():
    model = SentenceTransformer(params["normalizer_data"]["sim_model"])
    test_df = pd.read_csv(params["normalizer_data"]["test"])
    queries = test_df[params["normalizer_data"]["text_field"]]

    meddra_list = []
    meddra_dict = {}
    for line in open(params["normalizer_data"]["meddra"], 'r'):
        elems = line.split("$")
        ptid, text = int(elems[0]), elems[1]
        meddra_dict[text] = ptid
        meddra_list.append(text)

    print('calculating embeddings...')
    query_embeddings = model.encode(queries)
    sentence_embeddings = model.encode(meddra_list)
    print('embeddings done')

    scores = []
    for i in trange(len(queries)):
        scores_aux = []
        for j in range(len(meddra_list)):
            scores_aux.append((meddra_dict.get(meddra_list[j]),
                               meddra_list[j],
                               cosine_similarity(query_embeddings[i].reshape(1, -1),
                                                 sentence_embeddings[j].reshape(1, -1))[0][0]))

        scores_aux = max(scores_aux, key=lambda item: item[2])
        scores.append(scores_aux)

    scores_df = pd.DataFrame(scores, columns=['idmeddra', 'meddra', 'score'])
    scores_df.to_csv(params["normalizer_data"]["test_pred_scores_sim"], index=False)


def get_train_valid_and_test():
    master = pd.read_csv(params["normalizer_data"]["master"])
    train, valid, test = np.split(master.sample(frac=1), [int(0.7 * len(master)), int((0.15 + 0.7) * len(master))])
    train.to_csv(params["normalizer_data"]["train"], index=False)
    valid.to_csv(params["normalizer_data"]["valid"], index=False)
    test.to_csv(params["normalizer_data"]["test"], index=False)
