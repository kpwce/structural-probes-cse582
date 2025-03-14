import logging
import math
import os

import torch
import yaml
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, models, util, InputExample, evaluation
from transformers import get_linear_schedule_with_warmup, AdamW
from datasets import load_dataset, concatenate_datasets
from scipy import stats
import matplotlib.pyplot as plt


class TrainingArgs:
    def __init__(self, raw):
        self.root = raw['root']
        self.learning_rate = raw['learning_rate']
        self.do_train = raw['do_train']
        self.seeds = raw['seeds']
        self.epochs = raw['epochs']
        self.batch_size = raw['batch_size']
        self.save_path = raw['save_path']
        self.max_num_samples_train = raw['max_num_samples_train']
        self.max_num_samples_validation = raw['max_num_samples_validation']
        
class ReportingArgs:
    def __init__(self, raw):
        self.root = raw['root']
        self.csv_path = raw['csv_path']

class Config:
    def __init__(self, raw):
        self.training_args = TrainingArgs(raw['training_args'])
        self.reporting_args = ReportingArgs(raw['reporting'])
        self.dataset = raw['dataset']['root']
        self.encoders = raw['encoders']


with open('config/config.yaml', 'r') as f:
    args = Config(yaml.safe_load(f))

# encoder = args['model']['name']
results_save_path = os.path.join(args.reporting_args.root, args.reporting_args.csv_path)
seeds = args.training_args.seeds

device = torch.device("cuda")


def prep_data(dataset, norm=True) -> list[InputExample]:
    """
    Prepares dataset with pairs of sentences.

    Args:
        dataset (_type_): _description_
        norm (bool, optional): _description_. Defaults to True.

    Returns:
        list[InputExample]: Formats dataset as
                [   InputExample(texts=[sentence1, sentence2], label=0.8), 
                    InputExample(texts=[sentence1, sentence2], label=0.2)]
    """
    first_sent = [i['sentence1'] for i in dataset]
    second_sent = [i['sentence2'] for i in dataset]
    if norm:
        norm_labels = [i['similarity_score'] / 5.0 for i in dataset]
    else:
        norm_labels = [i['similarity_score'] for i in dataset]

    return [InputExample(texts=[str(x), str(y)], label=float(z)) for x, y, z in
            zip(first_sent, second_sent, norm_labels)]


def plot_df(df, encoder, spearman_rank, test):
    x = df.preds
    y = df.gold_labels
    plt.scatter(x, y)
    plt.plot(x.sort_values(), y.sort_values(), color='red')
    plt.title("Spearman correlation coefficient: {:.2f}".format(spearman_rank.statistic))
    plt.xlabel("predictions")
    plt.ylabel("gold labels")
    plt.savefig(results_save_path + f"spearman_rank_test_{encoder}_{test}")
    plt.show()

if __name__  == "__main__":
    lr = args.training_args.learning_rate
    if args.training_args.do_train:
        en_dataset = load_dataset("stsb_multi_mt", "en")
        es_dataset = load_dataset("stsb_multi_mt", "es")

        print(en_dataset)
        print(es_dataset)

        print(es_dataset['train'][100])
        print(en_dataset['train'][100])

        train_dataset = concatenate_datasets([es_dataset['train'], en_dataset['train']])
        dev_dataset = concatenate_datasets([es_dataset['dev'], en_dataset['dev']])
        test_dataset = concatenate_datasets([es_dataset['test'], en_dataset['test']])

        print(train_dataset)
        print(train_dataset[500])

        """## Prepare training and validation data split"""

        # if we use fit function to train, it tokenizes our data for us
        train_ds = prep_data(train_dataset)
        val_ds = prep_data(dev_dataset)

        # we split 90/10
        train_size = len(train_ds)
        val_size = len(val_ds)

        print('{:>5,} training samples'.format(train_size))
        print('{:>5,} validation samples'.format(val_size))

        epochs = int(args.training_args.epochs)
        batch_size = int(args.training_args.batch_size)

        # Put max number of training samples for debugging purposes.
        train_dataloader = DataLoader(
            train_ds,  # The training samples.
            num_workers=1,
            batch_size=batch_size,  # Use this batch size.
            shuffle=True  # Select samples randomly for each batch
        )

        val_dataloader = DataLoader(
            val_ds,
            num_workers=1,
            batch_size=batch_size,  # Use the same batch size
            shuffle=True
        )
        

        encoders = ['xlm-roberta-base']

        for encoder in encoders:
            # make model
            word_embedding_model = models.Transformer(encoder)

            # Apply mean pooling to get one fixed sized sentence vector
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                        pooling_mode_mean_tokens=True,
                                        pooling_mode_cls_token=False,
                                        pooling_mode_max_tokens=False)

            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

            train_loss = losses.CosineSimilarityLoss(model=model)
            evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_ds,
                                                                                    name='sts-dev')  # testing the model

            # evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(test_examples, name=f'{encoder}-sts-2e-5-lr')
            print(evaluator(model, output_path=results_save_path))
            
            # Configure the training. We skip evaluation in this example
            warmup_steps = math.ceil(len(train_dataloader) * epochs * 0.1)  # 10% of train data for warm-up
            logging.info("Warmup-steps: {}".format(warmup_steps))

            model_path = os.path.join(args.training_args.root, args.training_args.save_path,
                                    f'{encoder}_sts_fit_{lr}')

            model.fit(train_objectives=[(train_dataloader, train_loss)],
                      evaluator=evaluator,
                      epochs=epochs,
                      optimizer_params={'lr': float(lr)},
                      evaluation_steps=1000,
                      warmup_steps=warmup_steps,
                      output_path=model_path)

            # model = SentenceTransformer(model_path)
            test_examples = prep_data(test_dataset)
            test_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(test_examples,
                                                                                        name=f'{encoder}-sts-{lr}-lr')
            print(test_evaluator(model, output_path=results_save_path))