import numpy as np
from sklearn.metrics import confusion_matrix

def draw_con_matrix(path):
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    labels = ["O", "B-LOC", "B-PER", "B-ORG", "I-PER", "I-ORG", "B-MISC", "I-MISC", "I-LOC"]
    true_labels = []
    predicted_labels = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            token, true_tag, predicted_tag = parts
            true_labels.append(true_tag)
            predicted_labels.append(predicted_tag)

    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
    s = {}
    cm = np.array(conf_matrix)
    for i in range(len(labels)):
        label = labels[i]
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        if label == 'O':
            s[label] = [TP, FP, FN]
            continue
        l = label[2:]
        if s.get(l) is None:
            s[l] = [TP, FP, FN]
            continue
        s[l] = [s.get(l)[0] + TP, s.get(l)[1] + FP, s.get(l)[2] + FN]
    print(len(s))
    for k, v in s.items():
        TP = v[0]
        FP = v[1]
        FN = v[2]
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = (2 * precision * recall) / (precision + recall)
        print(f"For label '{k}':")
        print(f"TP: {TP}, FP: {FP}, FN: {FN}")
        print(f'precision: {precision}, recall: {recall}, f1: {f1}')
        print("--------------")


# processed 46435 tokens with 5648 phrases; found: 4808 phrases; correct: 3919.
# accuracy:  94.43%; precision:  81.51%; recall:  69.39%; FB1:  74.96
#               LOC: precision:  82.72%; recall:  81.24%; FB1:  81.97  1638
#              MISC: precision:  81.00%; recall:  64.39%; FB1:  71.75  558
#               ORG: precision:  75.07%; recall:  61.65%; FB1:  67.70  1364
#               PER: precision:  87.18%; recall:  67.29%; FB1:  75.95  1248



draw_con_matrix('result.txt')