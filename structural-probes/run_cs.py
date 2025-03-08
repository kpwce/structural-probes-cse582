"""Generate data for CSE 582 project.
cat code-switched-2024/data/semantics/cs_majEn_semantic.csv | python structural-probes/run_cs.py code-switched-2024/codeswitch.yaml

Tiny testing file:
cat code-switched-2024/data/parallel_Spanglish_tiny.csv | python structural-probes/run_cs.py code-switched-2024/codeswitch.yaml
"""
import csv
import os
import sys
import string
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
from convert_dp_to_trees import filter_subtree_edges, get_edit_distance
from language_alignment import get_alignment_2, get_lang_subintervals


def write_distance_csv(args, words, prediction, uid):
    """Writes a distance matrix to CSV.

    Args:
        args: yaml config
        words: list of strings representing the sentence
        prediction: numpy matrix of shape (len(words), len(words))
        uid: index for this sentence, for use in the CSV filename.
    """
    csv_filename = os.path.join(args['reporting']['root'], f'demo-dist-pred{uid}.csv')
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header (words)
        writer.writerow([''] + words)
        for i, word in enumerate(words):
            writer.writerow([word] + prediction[i].tolist())

def write_depth_csv(args, words, prediction, uid):
    """Writes a depth prediction to CSV.

    Args:
        args: yaml config dictionary
        words: list of strings representing the sentence
        prediction: numpy array of shape (len(words),)
        uid: index for identifying this sentence, for use in the CSV filename.
    """
    csv_filename = os.path.join(args['reporting']['root'], f'demo-depth-pred{uid}.csv')

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Word', 'Predicted Depth'])
        for word, pred in zip(words, prediction):
            writer.writerow([word, pred])

def write_edges_csv(args, words, prediction, uid):
    csv_filename = os.path.join(args['reporting']['root'], f'demo-edges-pred{uid}.csv')

    predicted_edges = reporter.prims_matrix_to_edges(prediction, words, words)
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Start', 'End'])
        # Write each edge in the format (source, target, weight)
        for edge in predicted_edges:
            writer.writerow(edge)

        # no need to print picture
        # print_tikz(args, predicted_edges, untokenized_sent)
    
    return predicted_edges

def probe_line(args, line, uid, tokenizer, model, distance_probe, depth_probe):
    
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
        write_distance_csv(args, untokenized_sent, distance_predictions, uid)
        write_depth_csv(args, untokenized_sent, depth_predictions, uid)

        # You can also print or store the predicted edges if necessary
        # TODO Alysa/David: figure out what format we want these edges
        
        word_to_id = {word: i for i, word in enumerate(untokenized_sent)}
        
        # probably should factor this out into another function but leaving this here for now
        edges = write_edges_csv(args, untokenized_sent, distance_predictions, uid)
    
    return edges, word_to_id


def report_on_stdin(args, file_path):
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

    # for index, line in tqdm(enumerate(sys.stdin), desc='[demoing]'):
    #     reader = csv.reader([line])
    
    with open(file_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for index, row in tqdm(enumerate(reader)):
            cs_line = row[2]
            en_line = row[3]
            es_line = row[4]
            subtree_classification, all_langs = get_lang_subintervals(cs_line)
            en_list = get_lang_subintervals(en_line)
            es_list = get_lang_subintervals(es_line)
            
            cs_sentence_from_trees = ""
            for tree in subtree_classification:
                cs_sentence_from_trees += ' '.join(tree)

            cs_edges, cs_word_to_id = probe_line(args, cs_line, f"{index}_cs", tokenizer, model, distance_probe, depth_probe)
            en_edges, en_word_to_id = probe_line(args, en_line, f"{index}_en", tokenizer, model, distance_probe, depth_probe) 
            es_edges, es_word_to_id = probe_line(args, es_line, f"{index}_es", tokenizer, model, distance_probe, depth_probe)
            
            print(f"Base line: en-en: {get_edit_distance(en_edges, en_edges)}, cs-cs: {get_edit_distance(cs_edges, cs_edges)}, es-es: {get_edit_distance(es_edges, es_edges)}")
            print("Edit distance (cs-en) %d".format(get_edit_distance(cs_edges, en_edges)))
            print("Edit distance (cs-es) %d".format(get_edit_distance(cs_edges, es_edges)))
            print("Edit distance (en-es) %d".format(get_edit_distance(en_edges, es_edges)))
            
            # get alignment for each subtree and their GED
            for i, cs_target_list in enumerate(subtree_classification):
                matching_en_list = []
                if all_langs[i] == "en":
                    matching_en_list = get_alignment_2(cs_target_list, en_list)
                    mono_subtree_edges = filter_subtree_edges(matching_en_list, en_edges, en_word_to_id)
                if all_langs[i] == "spa":
                    matching_es_list = get_alignment_2(cs_target_list, es_list)
                    mono_subtree_edges = filter_subtree_edges(matching_es_list, es_edges, es_word_to_id)
                else:
                    raise Exception(f"{all_langs[i]} Language not found")
                cs_subtree_edges = filter_subtree_edges(cs_target_list, cs_edges, cs_word_to_id)
                print(f"Subtree '{' '.join(cs_target_list)}' GED: {get_edit_distance(cs_subtree_edges, mono_subtree_edges)}")


if __name__ == '__main__':
    file_path = "code-switched-2024/data/parallel_Spanglish_tiny.csv"
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
    report_on_stdin(yaml_args, file_path)
