import argparse
from utils import *
from transformers import *

def output_gold(dir, split, suffix):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)
    data = load_data(dir, "individual_%s" % split, suffix)
    features = convert_to_features_roberta(data, tokenizer, max_length=178,
                                           evaluation=True, instance=False, end_to_end=True)
    labels = select_field(features, 'label')

    # collect unique question ids for EM calculation
    question_ids = select_field(features, 'question_id')

    # collect unique question culster for EM-cluster calculation
    question_cluster = select_field(features, 'question_cluster')
    question_cluster_size = select_field(features, 'cluster_size')
    idv_answers = select_field(features, 'individual_answers')

    # questions
    questions = select_field(features, 'question')

    gold = {}
    for id, l, q, c, cs, idv in zip(question_ids, labels, questions, question_cluster,
                                    question_cluster_size, idv_answers):
        gold[id] = {'label': l,
                    'cluster': c,
                    'cluster_size': cs,
                    'idv_answers': idv}
        if len(gold) < 3:
            print(gold)

    with open("./output/%s_gold.json" % (split + "_" + suffix.split(".")[0]), 'w') as outfile:
        json.dump(gold, outfile)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate TORQUE predictions')
    # Required Parameters
    parser.add_argument('--data_dir', type=str, help='data dir', default="data/")
    parser.add_argument('--split', type=str, help='dev/test', default="dev")
    parser.add_argument('--file_suffix', type=str, help='suffix', default="_end2end_final.json")
    
    args = parser.parse_args()

    output_gold(args.data_dir, args.split, args.file_suffix)