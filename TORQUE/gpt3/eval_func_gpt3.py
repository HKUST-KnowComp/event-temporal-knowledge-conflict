from tqdm import tqdm
from collections import Counter, defaultdict
def exact_match(question_ids, labels, predictions):
    em = defaultdict(list)
    for q, l, p in zip(question_ids, labels, predictions):
        em[q].append(l == p)
    print("Total %s questions" % len(em))
    return float(sum([all(v) for v in em.values()])) / float(len(em))

def cal_f1(pred_labels, true_labels, label_map, log=False):
    def safe_division(numr, denr, on_err=0.0):
        return on_err if denr == 0.0 else numr / denr

    assert len(pred_labels) == len(true_labels)

    num_tests = len(true_labels)

    total_true = Counter(true_labels)
    total_pred = Counter(pred_labels)

    labels = list(label_map)

    n_correct = 0
    n_true = 0
    n_pred = 0

    label_map
    # we only need positive f1 score
    exclude_labels = ['Negative']
    for label in labels:
        if label not in exclude_labels:
            true_count = total_true.get(label, 0)
            pred_count = total_pred.get(label, 0)

            n_true += true_count
            n_pred += pred_count

            correct_count = len([l for l in range(len(pred_labels))
                                 if pred_labels[l] == true_labels[l] and pred_labels[l] == label])
            n_correct += correct_count
    if log:
        logger.info("Correct: %d\tTrue: %d\tPred: %d" % (n_correct, n_true, n_pred))
    precision = safe_division(n_correct, n_pred)
    recall = safe_division(n_correct, n_true)
    f1_score = safe_division(2.0 * precision * recall, precision + recall)
    if log:
        logger.info("Overall Precision: %.4f\tRecall: %.4f\tF1: %.4f" % (precision, recall, f1_score))
    return f1_score


def evaluate(pred_dict, data_dict):
    
    def parse_results(text):
        words = [w.lower().strip().replace(".", "") for w in text.split(",")]
        return list(set(words))

    parsed_tokens = [parse_results(res) for key, res in pred_dict.items()]

    # convert the parsed tokens into one-hot predictions
    import numpy as np
    preds = []
    labels = []
    eval_idv_answers = []
    question_cluster_size = []
    question_cluster = []
    question_ids = []

    for (key, item), ans in tqdm(zip(data_dict.items(), parsed_tokens)):
        preds.append([1 if t.lower() in ans and t != "none" else 0 for t in item['context']])
        labels.append(item["answers"]["labels"])
        eval_idv_answers.append([a['labels'] for a in item['individual_answers']])
        question_cluster_size.append(item['cluster_size'])
        question_cluster.append(item["question_cluster"])
        question_ids.append(key)
    question_ids = [q for i, q in enumerate(question_ids) for x in range(len(labels[i]))]

    from collections import Counter, defaultdict



    label_map = {0: 'Negative', 1: 'Positive'}
    eval_loss, eval_accuracy, nb_eval_examples, nb_eval_steps = 0.0, 0.0, 0, 0
    all_preds, all_golds, max_f1s, macro_f1s = [], [], [], []
    f1_dist = defaultdict(list)
    em_counter = 0
    em_cluster_agg, em_cluster_relaxed, f1_cluster_80 = {}, {}, {}

    for idx in range(len(data_dict)):

        pred = preds[idx]
        all_preds.extend(pred)
        label = labels[idx]
        all_golds.extend(label)
        pred_names = [label_map[p] for p in pred]
        gold_names = [label_map[l] for l in label]
        is_em = (pred_names == gold_names)

        if sum(label) == 0 and sum(pred) == 0:
            macro_f1s.append(1.0)
        else:
            macro_f1s.append(cal_f1(pred_names, gold_names, {v:k for k,v in label_map.items()}))

        max_f1, instance_matched = 0, 0
        for gold in eval_idv_answers[idx]:
            label_names = [label_map[l] for l in gold]
            if pred_names == label_names: instance_matched = 1
            if sum(gold) == 0 and sum(pred) == 0:
                f1 = 1.0
            else:
                f1 = cal_f1(pred_names, label_names, {v:k for k,v in label_map.items()})
            # if f1 > max_f1: max_f1 = f1
            if f1 >= max_f1:
                max_f1 = f1
                key = len(gold)

        if question_cluster_size[idx] > 1:
            if question_cluster[idx] not in em_cluster_agg:
                em_cluster_agg[question_cluster[idx]] = 1
            if is_em == 0: em_cluster_agg[question_cluster[idx]] = 0

            if question_cluster[idx] not in em_cluster_relaxed:
                em_cluster_relaxed[question_cluster[idx]] = 1
            if instance_matched == 0: em_cluster_relaxed[question_cluster[idx]] = 0

            if question_cluster[idx] not in f1_cluster_80:
                f1_cluster_80[question_cluster[idx]] = 1
            if max_f1 < 0.8: f1_cluster_80[question_cluster[idx]] = 0

        max_f1s.append(max_f1)
        em_counter += instance_matched
        f1_dist[key].append(max_f1)

    assert len(all_preds) == len(question_ids)
    assert len(f1_cluster_80) == len(em_cluster_agg) 

    # em = exact_match(question_ids, all_golds, all_preds)
    eval_accuracy = eval_accuracy / len(all_preds)
    label_names = [label_map[l] for l in all_golds]
    pred_names = [label_map[p] for p in all_preds]
    # eval_pos_f1 = cal_f1(pred_names, label_names, {v:k for k,v in label_map.items()})

    em_cluster_relaxed_res = sum(em_cluster_relaxed.values()) / len(em_cluster_relaxed)
    em_cluster_agg_res = sum(em_cluster_agg.values()) / len(em_cluster_agg)
    f1_cluster_80_res = sum(f1_cluster_80.values()) / len(f1_cluster_80)

    label_names = [label_map[l] for l in all_golds]
    pred_names = [label_map[p] for p in all_preds]

    em = exact_match(question_ids, label_names, pred_names)
    eval_pos_f1 = cal_f1(pred_names, label_names, {v:k for k,v in label_map.items()})


    print(f"Eval on the current eval positive class Micro F1 (Agg) is: %.4f" % eval_pos_f1)
    print(f"Eval on the current eval positive class Macro F1 (Relaxed) is: %.4f" % np.mean(max_f1s)) # output F1
    print(f"Eval on the current eval positive class Macro F1 (Agg) is: %.4f" % np.mean(macro_f1s))

    print(f"Eval on the current eval exact match (Agg) ratio is: %.4f" % em)
    print(f"Eval on the current eval exact match ratio (Relaxed) is: %.4f" % (em_counter / len(data_dict))) # output EM

    print(f"Eval on %d Clusters" % len(em_cluster_relaxed))
    print(f"Eval on the current eval clustered EM (Agg) is: %.4f" % (em_cluster_agg_res))
    print(f"Eval on the current eval clustered EM (Relaxed) is: %.4f" % (em_cluster_relaxed_res))
    print(f"Eval on the current eval clusrered F1 (max>=0.8) is: %.4f" % (f1_cluster_80_res)) # consistency
