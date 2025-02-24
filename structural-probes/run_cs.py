"""Generate data for CSE 582 project.
cat code-switched-2024/data/semantics/cs_majEn_semantic.csv | python structural-probes/run_cs.py code-switched-2024/codeswitch.yaml
"""
import csv
import os
import sys
from argparse import ArgumentParser

import data
import numpy as np
import probe
import reporter
import run_experiment
import torch
import yaml
from pytorch_pretrained_bert import BertModel, BertTokenizer
from tqdm import tqdm


def write_distance_csv(args, words, prediction, sent_index):
    """Writes a distance matrix to CSV.

    Args:
        args: yaml config
        words: list of strings representing the sentence
        prediction: numpy matrix of shape (len(words), len(words))
        sent_index: index for this sentence, for use in the CSV filename.
    """
    csv_filename = os.path.join(args['reporting']['root'], f'demo-dist-pred{sent_index}.csv')
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header (words)
        writer.writerow([''] + words)
        for i, word in enumerate(words):
            writer.writerow([word] + prediction[i].tolist())


def write_depth_csv(args, words, prediction, sent_index):
    """Writes a depth prediction to CSV.

    Args:
        args: yaml config dictionary
        words: list of strings representing the sentence
        prediction: numpy array of shape (len(words),)
        sent_index: index for identifying this sentence, for use in the CSV filename.
    """
    csv_filename = os.path.join(args['reporting']['root'], f'demo-depth-pred{sent_index}.csv')

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Word', 'Predicted Depth'])
        for word, pred in zip(words, prediction):
            writer.writerow([word, pred])


def report_on_stdin(args):
    """Runs a trained structural probe on sentences piped to stdin.

    Sentences should be space-tokenized.
    A single distance matrix and depth data will be written for each line of stdin to CSV.

    Args:
        args: the yaml config dictionary
    """

    # Define the BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    model = BertModel.from_pretrained('bert-large-cased')
    LAYER_COUNT = 24
    FEATURE_COUNT = 1024
    model.to(args['device'])
    model.eval()

    # Define the distance probe
    distance_probe = probe.TwoWordPSDProbe(args)
    distance_probe.load_state_dict(torch.load(args['probe']['distance_params_path'], map_location=args['device']))

    # Define the depth probe
    depth_probe = probe.OneWordPSDProbe(args)
    depth_probe.load_state_dict(torch.load(args['probe']['depth_params_path'], map_location=args['device']))

    for index, line in tqdm(enumerate(sys.stdin), desc='[demoing]'):
        # Tokenize the sentence and create tensor inputs to BERT
        untokenized_sent = line.strip().split()
        tokenized_sent = tokenizer.wordpiece_tokenizer.tokenize('[CLS] ' + ' '.join(line.strip().split()) + ' [SEP]')
        untok_tok_mapping = data.SubwordDataset.match_tokenized_to_untokenized(tokenized_sent, untokenized_sent)

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sent)
        segment_ids = [1 for x in tokenized_sent]

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segment_ids])

        tokens_tensor = tokens_tensor.to(args['device'])
        segments_tensors = segments_tensors.to(args['device'])

        with torch.no_grad():
            # Run sentence tensor through BERT after averaging subwords for each token
            encoded_layers, _ = model(tokens_tensor, segments_tensors)
            single_layer_features = encoded_layers[args['model']['model_layer']]
            representation = torch.stack([torch.mean(single_layer_features[0, untok_tok_mapping[i][0]:untok_tok_mapping[i][-1] + 1, :], dim=0) for i in range(len(untokenized_sent))], dim=0)
            representation = representation.view(1, *representation.size())

            # Run BERT token vectors through the trained probes
            distance_predictions = distance_probe(representation.to(args['device'])).detach().cpu()[0][:len(untokenized_sent), :len(untokenized_sent)].numpy()
            depth_predictions = depth_probe(representation).detach().cpu()[0][:len(untokenized_sent)].numpy()

            # Write results to CSV (for now)
            # Might not need since we want edges for graph edit distance
            write_distance_csv(args, untokenized_sent, distance_predictions, index)
            write_depth_csv(args, untokenized_sent, depth_predictions, index)

            # You can also print or store the predicted edges if necessary
            # TODO Alysa/David: figure out what format we want these edges
            # probably should factor this out into another function but leaving this here for now
            predicted_edges = reporter.prims_matrix_to_edges(distance_predictions, untokenized_sent, untokenized_sent)
            with open("TODO_name_this.csv", mode='w', newline='', encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(['Start', 'End', 'Weight'])

                # Write each edge in the format (source, target, weight)
                for edge in predicted_edges:
                    writer.writerow(edge)

            # no need to print picture
            # print_tikz(args, predicted_edges, untokenized_sent)


if __name__ == '__main__':
    argp = ArgumentParser()
    argp.add_argument('experiment_config')
    argp.add_argument('--results-dir', default='',
                      help='Set to reuse an old results dir; '
                           'if left empty, new directory is created')
    argp.add_argument('--seed', default=0, type=int,
                      help='sets all random seeds for (within-machine) reproducibility')
    cli_args = argp.parse_args()
    if cli_args.seed:
        np.random.seed(cli_args.seed)
        torch.manual_seed(cli_args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    yaml_args = yaml.load(open(cli_args.experiment_config), Loader=yaml.Loader)
    run_experiment.setup_new_experiment_dir(cli_args, yaml_args, cli_args.results_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    yaml_args['device'] = device
    report_on_stdin(yaml_args)
