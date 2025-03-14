import os
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer, util, InputExample, evaluation
from datasets import load_dataset, concatenate_datasets, Dataset
import argparse
import matplotlib.pyplot as plt
import random

SEED=100
random.seed(SEED)
# random.seed(100)


def get_args():
    parser = argparse.ArgumentParser(description='Semantic Text Similarity Evaluation')
    parser.add_argument('--dataset', type=str, required=True, help='Name of dataset in dataset root dir')
    parser.add_argument('--encoder', type=str, default='xlm-roberta-base', help='Name of encoder to use')
    # 'bert-base-multilingual-uncased'
    parser.add_argument('--model_dir', type=str, required=True, help='Path to save the model')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data shuffle')
    parser.add_argument('--max_num_samples_eval', type=int, default=6000, help='Maximum number of samples used in evaluation')

    return parser.parse_args()


def prep_data(dataset, col1, col2, norm=True) -> list[InputExample]:
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
    first_sent = [i[col1] for i in dataset]
    second_sent = [i[col2] for i in dataset]
    if 'similarity_score' in dataset:
        if norm:
            norm_labels = [i['similarity_score'] / 5.0 for i in dataset]
        else:
            norm_labels = [i['similarity_score'] for i in dataset]
    else:
        norm_labels = [0 for _ in dataset]

    return [InputExample(texts=[str(x), str(y)], label=float(z)) for x, y, z in
            zip(first_sent, second_sent, norm_labels)]
    
    
def test_cs_experiments(list_df, lang1, lang2, results_save_path):
    list_preds = []
    for i, df in enumerate(list_df):
        first_sent = [df['sentence1'][j] for j in range(len(df))]
        second_sent = [df['sentence2'][j] for j in range(len(df))]

        first_sent = model.encode(first_sent, convert_to_tensor=True)
        second_sent = model.encode(second_sent, convert_to_tensor=True)
        pred_scores = util.cos_sim(first_sent, second_sent).detach().cpu().numpy().tolist()
        pred_scores = [pred_scores[k][k] for k in range(len(first_sent))]
        print(type(pred_scores), pred_scores)
        list_preds.append(pred_scores)

    # Once we have two pred scores (one for each data we pair against each other)
    # get spearman rank correlation
    print(list_preds[0], print(len(list_preds[0])))
    print('length of preds list is', len(list_preds))
    preds = list_preds[0]
    gold = list_preds[1]  # not real gold, just other preds cuz we compare against each other
    spearman_rank = stats.spearmanr(preds, gold)
    print(spearman_rank.statistic)

    outputs_df = pd.DataFrame({'preds': preds,
                               'gold_labels': gold})

    outputs_df.to_csv(os.path.join(results_save_path, f'sts_model_{encoder}_{lang1}_{lang2}_preds.csv'))
    plot_df(outputs_df, encoder, spearman_rank, f'{lang1}_vs_{lang2}', results_save_path)

    # to get score from evaluator using sentbert library
    # 1. create dataset in right format
    # add pred scores to original df so that we can use later of sentbert eval
    list_df[0]['scores'] = list_preds[1]    # we assing the opposite pred scores as gold labels b/c we want to compare model output for this
    list_df[1]['scores'] = list_preds[0]

    # prep data
    first_sent = [df['sentence1'][j] for j in range(len(list_df[0]))]
    second_sent = [df['sentence2'][j] for j in range(len(list_df[0]))]
    norm_labels = [df['scores'][j] for j in range(len(list_df[1]))]   # opposite scores b/c we compare model gen against this
    examples = [InputExample(texts=[str(x), str(y)], label=float(z)) for x, y, z in
            zip(first_sent, second_sent, norm_labels)]

    # 2. define test evaluator
    test_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(examples, name=f'sts-{lang1}_{lang2}_{lr}')
    test_evaluator(model, output_path=results_save_path)


def get_similarity_scores(model, dataset, col1: str = "sentence1", col2: str = "sentence2", output_dir: str = ""):
    """
    Saves similarity scores between sentence pairs in the dataset to results_save_path.

    Args:
        model (_type_): _description_
        dataset (_type_): _description_
        col1 (str): name of first column
        col2 (str): name of second column
        output_dir (str): _description_
    """
    test_dataset = dataset
    test_examples = prep_data(test_dataset, col1, col2)
    
    test_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(test_examples)
    test_evaluator(model, output_path=output_dir)

    first_sent = [i[col1] for i in test_dataset]
    second_sent = [i[col2] for i in test_dataset]
    # norm_labels = [i['similarity_score'] / 5.0 for i in test_dataset] if 'similarity_score' in test_dataset else None

    first_sent = model.encode(first_sent, convert_to_tensor=True)
    second_sent = model.encode(second_sent, convert_to_tensor=True)
    pred_scores = model.similarity(first_sent, second_sent).tolist()
    # sim_scores = norm_labels

    preds = [pred_scores[i][i] for i in range(len(first_sent))]
    
    outputs = {'preds': preds}
    # if sim_scores is not None:
    #     outputs['gold_labels'] = sim_scores
        
    outputs_df = pd.DataFrame(outputs)
    outputs_df.to_csv(os.path.join(output_dir, f'sts_model_preds_{col1}_{col2}.csv'))
    
    return preds


def get_spearman_rank_correlation(scores1, scores2, output_dir, title=""):
    # get spearman rank correlation
    args = get_args()
    encoder = args.encoder
    
    spearman_rank = stats.spearmanr(scores1, scores2)
    print("Spearman rank correlation", spearman_rank.statistic)

    x = scores1
    y = scores2
    plt.scatter(x, y)
    plt.plot(sorted(x), sorted(x), color='red')
    plt.title("Spearman correlation coefficient: {:.2f}".format(spearman_rank.statistic) + f"\n{title}")
    plt.xlabel("sentence1")
    plt.ylabel("sentence2")
    plt.savefig(os.path.join(output_dir, f"spearman_rank_test_{encoder}_{SEED}"))
    plt.show()


def get_random_sentence_pairs(dataset, cols: list[str]):
    """
    Given a dataset with columns containing cols, return a list of datasets where 
    each dataset has two columns 'sentence1' and 'sentence2' of random sentence
    pairs using sentences from dataset[col] for col in cols. 
    
    Each dataset has the same ordering.

    Args:
        dataset (_type_): _description_
    """
    new_datasets = []
    n = len(dataset)  # Assumes all datasets have the same length
    s1_order = random.sample(range(n), n // 2)
    s2_order = random.sample(range(n), n // 2)
    for col in cols:
        sentence_pairs = {
            'sentence1': [dataset[col][i] for i in s1_order], 
            'sentence2': [dataset[col][i] for i in s2_order]
        }
        new_datasets.append(Dataset.from_dict(sentence_pairs))
    return new_datasets

if __name__  == "__main__":
    args = get_args()
    
    seed = args.seed
    dataset_file = args.dataset
    max_num_samples_eval = args.max_num_samples_eval
    encoder = args.encoder

    data = pd.read_csv(dataset_file)

    # Shuffle the dataset.
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)[:max_num_samples_eval]
    print(data.head())

    # All the columns are cs, es, en
    cs_maj_en = Dataset.from_pandas(pd.read_csv('data/semantics/cs_majEn_semantic.csv'))
    cs_maj_es = Dataset.from_pandas(pd.read_csv('data/semantics/cs_majEs_semantic.csv'))
    cs_random = Dataset.from_pandas(pd.read_csv('data/semantics/cs_random_semantic.csv'))
    
    # Majority en cs
    # - scores cs with en, cs with es
    # Majority es cs
    # - scores cs with en, cs with cs
    # Get correlation with en with es
    
    en_dataset = load_dataset("stsb_multi_mt", "en")
    es_dataset = load_dataset("stsb_multi_mt", "es")

    print(en_dataset)
    print(es_dataset)

    # test partition of train sts dataset/
    dataset = concatenate_datasets([es_dataset['test'], en_dataset['test']])

    model_path = os.path.join(
        args.model_dir,
        f'{encoder}_sts_fit_2e-05'
    )
    model = SentenceTransformer(model_path)
    
    # cs_en_scores = get_similarity_scores(model, cs_maj_en, "cs_sentences", "en_sentences", args.output_dir)
    # cs_es_scores = get_similarity_scores(model, cs_maj_en, "cs_sentences", "es_sentences", args.output_dir)
    
    # en_es_scores = get_similarity_scores(model, cs_maj_en, "en_sentences", "es_sentences", args.output_dir)
    # cs_en_scores = get_similarity_scores(model, cs_maj_en, "cs_sentences", "en_sentences", args.output_dir)
    # get_spearman_rank_correlation(en_es_scores, cs_en_scores, args.output_dir)
    
    # en_es_scores = get_similarity_scores(model, cs_maj_en, "en_sentences", "es_sentences", args.output_dir)
    # cs_es_scores = get_similarity_scores(model, cs_maj_en, "cs_sentences", "es_sentences", args.output_dir)
    # get_spearman_rank_correlation(en_es_scores, cs_es_scores, args.output_dir)
    
    # sentence_pairs = get_random_sentence_pairs(cs_maj_en, ["en_sentences", "es_sentences"])
    # en_pairs, es_pairs = sentence_pairs
    # en_scores = get_similarity_scores(model, en_pairs, "sentence1", "sentence2", args.output_dir)
    # es_scores = get_similarity_scores(model, es_pairs, "sentence1", "sentence2", args.output_dir)
    # get_spearman_rank_correlation(en_scores, es_scores, args.output_dir)
    
    # print("Maj en, en, es")
    # sentence_pairs = get_random_sentence_pairs(cs_maj_en, ["en_sentences", "es_sentences"])
    # en_pairs, es_pairs = sentence_pairs
    # en_scores = get_similarity_scores(model, en_pairs, "sentence1", "sentence2", args.output_dir)
    # es_scores = get_similarity_scores(model, es_pairs, "sentence1", "sentence2", args.output_dir)
    # get_spearman_rank_correlation(en_scores, es_scores, args.output_dir)
    
    # print("Maj en, en, cs")
    # sentence_pairs = get_random_sentence_pairs(cs_maj_en, ["en_sentences", "cs_sentences"])
    # en_pairs, cs_pairs = sentence_pairs
    # en_scores = get_similarity_scores(model, en_pairs, "sentence1", "sentence2", args.output_dir)
    # cs_scores = get_similarity_scores(model, cs_pairs, "sentence1", "sentence2", args.output_dir)
    # get_spearman_rank_correlation(en_scores, cs_scores, args.output_dir)

    # print("Maj en, es, cs")
    # sentence_pairs = get_random_sentence_pairs(cs_maj_en, ["es_sentences", "cs_sentences"])
    # a, b = sentence_pairs
    # a_s = get_similarity_scores(model, a, "sentence1", "sentence2", args.output_dir)
    # b_s = get_similarity_scores(model, b, "sentence1", "sentence2", args.output_dir)
    # get_spearman_rank_correlation(a_s, b_s, args.output_dir)
    
    # print("Maj es, en, es")
    # sentence_pairs = get_random_sentence_pairs(cs_maj_es, ["en_sentences", "es_sentences"])
    # a, b = sentence_pairs
    # a_s = get_similarity_scores(model, a, "sentence1", "sentence2", args.output_dir)
    # b_s = get_similarity_scores(model, b, "sentence1", "sentence2", args.output_dir)
    # get_spearman_rank_correlation(a_s, b_s, args.output_dir)
    
    print("Maj es, en, cs")
    sentence_pairs = get_random_sentence_pairs(cs_maj_es, ["en_sentences", "cs_sentences"])
    a, b = sentence_pairs
    a_s = get_similarity_scores(model, a, "sentence1", "sentence2", args.output_dir)
    b_s = get_similarity_scores(model, b, "sentence1", "sentence2", args.output_dir)
    get_spearman_rank_correlation(a_s, b_s, args.output_dir, "Correlation between en & cs in a majority cs-es corpus")
    
    # print("Maj es, es, cs")
    # sentence_pairs = get_random_sentence_pairs(cs_maj_es, ["es_sentences", "cs_sentences"])
    # a, b = sentence_pairs
    # a_s = get_similarity_scores(model, a, "sentence1", "sentence2", args.output_dir)
    # b_s = get_similarity_scores(model, b, "sentence1", "sentence2", args.output_dir)
    # get_spearman_rank_correlation(a_s, b_s, args.output_dir)
    
    # print("Random, en, es")
    # sentence_pairs = get_random_sentence_pairs(cs_random, ["en_sentences", "es_sentences"])
    # en_pairs, es_pairs = sentence_pairs
    # en_scores = get_similarity_scores(model, en_pairs, "sentence1", "sentence2", args.output_dir)
    # es_scores = get_similarity_scores(model, es_pairs, "sentence1", "sentence2", args.output_dir)
    # get_spearman_rank_correlation(en_scores, es_scores, args.output_dir)
    
    # print("Random, en, cs")
    # sentence_pairs = get_random_sentence_pairs(cs_random, ["en_sentences", "cs_sentences"])
    # en_pairs, cs_pairs = sentence_pairs
    # en_scores = get_similarity_scores(model, en_pairs, "sentence1", "sentence2", args.output_dir)
    # cs_scores = get_similarity_scores(model, cs_pairs, "sentence1", "sentence2", args.output_dir)
    # get_spearman_rank_correlation(en_scores, cs_scores, args.output_dir)
        
    # print("Random, es, cs")
    # sentence_pairs = get_random_sentence_pairs(cs_random, ["es_sentences", "cs_sentences"])
    # es_pairs, cs_pairs = sentence_pairs
    # es_scores = get_similarity_scores(model, es_pairs, "sentence1", "sentence2", args.output_dir)
    # cs_scores = get_similarity_scores(model, cs_pairs, "sentence1", "sentence2", args.output_dir)
    # get_spearman_rank_correlation(es_scores, cs_scores, args.output_dir)
