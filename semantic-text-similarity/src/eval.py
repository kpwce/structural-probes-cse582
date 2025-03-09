import os
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer, util, InputExample, evaluation
from datasets import load_dataset, concatenate_datasets
import argparse
import matplotlib.pyplot as plt


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


def plot_df(df, encoder, spearman_rank, test, results_save_path):
    x = df.preds
    y = df.gold_labels
    plt.scatter(x, y)
    plt.plot(x.sort_values(), y.sort_values(), color='red')
    plt.title("Spearman correlation coefficient: {:.2f}".format(spearman_rank.statistic))
    plt.xlabel("predictions")
    plt.ylabel("gold labels")
    plt.savefig(os.path.join(results_save_path, f"spearman_rank_test_{encoder}_{test}"))
    plt.show()
    
    
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

    outputs_df.to_csv(os.path.join(results_save_path, f'sts_model_{encoder}_{lang1}_{lang2}_{lr}_preds.csv'))
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


def test_after_train(model, encoder, results_save_path):
    # test partition of train sts dataset/
    en_dataset = load_dataset("stsb_multi_mt", "en")
    es_dataset = load_dataset("stsb_multi_mt", "es")

    print(en_dataset)
    print(es_dataset)

    print(es_dataset['train'][100])
    print(en_dataset['train'][100])

    test_dataset = concatenate_datasets([es_dataset['test'], en_dataset['test']])
    test_examples = prep_data(test_dataset)

    breakpoint()
    
    test_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(test_examples)
    test_evaluator(model, output_path=results_save_path)

    first_sent = [i['sentence1'] for i in test_dataset]
    second_sent = [i['sentence2'] for i in test_dataset]
    norm_labels = [i['similarity_score'] / 5.0 for i in test_dataset]

    first_sent = model.encode(first_sent, convert_to_tensor=True)
    second_sent = model.encode(second_sent, convert_to_tensor=True)
    pred_scores = model.similarity(first_sent, second_sent).tolist()
    sim_scores = [score for score in norm_labels]

    preds = [pred_scores[i][i] for i in range(len(first_sent))]

    # get spearman rank correlation
    spearman_rank = stats.spearmanr(preds, sim_scores)
    print(spearman_rank.statistic)

    outputs_df = pd.DataFrame({'preds': preds,
                               'gold_labels': sim_scores})

    outputs_df.to_csv(os.path.join(results_save_path, f'sts_model_{encoder}_preds.csv'))
    plot_df(outputs_df, encoder, spearman_rank, "idk", results_save_path)


def test_fn(model, encoder, dataset, output_path):
    # Get embeddings of sentences.
    first_sent = dataset['sentence1'].tolist()
    second_sent = dataset['sentence2'].tolist()
    norm_labels = [i['similarity_score'] / 5.0 for i in test_dataset]

    first_sent = model.encode(first_sent, convert_to_tensor=True)
    second_sent = model.encode(second_sent, convert_to_tensor=True)
    pred_scores = model.similarity(first_sent, second_sent).tolist()
    sim_scores = [score for score in norm_labels]
    
    # Test evaluator.
    test_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(test_examples)
    test_evaluator(model, output_path=output_path)
    

if __name__  == "__main__":
    # cs experiment examples
    parser = argparse.ArgumentParser(description='Semantic Text Similarity Evaluation')
    parser.add_argument('--dataset', type=str, required=True, help='Name of dataset in dataset root dir')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to save the model')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data shuffle')
    parser.add_argument('--max_num_samples_eval', type=int, default=6000, help='Maximum number of samples used in evaluation')

    args = parser.parse_args()
    
    seed = args.seed
    dataset_file = args.dataset
    max_num_samples_eval = args.max_num_samples_eval

    data = pd.read_csv(dataset_file)

    # Shuffle the dataset.
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)[:max_num_samples_eval]
    print(data.head())


    encoders = ['xlm-roberta-base'] # 'bert-base-multilingual-uncased', 
    for encoder in encoders:
        model_path = os.path.join(
            args.model_dir,
            f'{encoder}_sts_fit_2e-6'
        )
        model = SentenceTransformer(model_path)
        test_after_train(model, encoder, args.output_dir)
        # test_cs_experiments([cs_cs, es_es], 'cs_cs', 'es_es', lr=learning_rate)
        # test_cs_experiments([cs_cs, en_en], 'cs_cs', 'en_en', lr=learning_rate)

        # test_cs_experiments([en_es, cs_es], 'en_es', f'cs_es{encoder}', lr=learning_rate)
        # test_cs_experiments([en_es, cs_en], 'en_es', f'cs_en{encoder}', lr=learning_rate)
