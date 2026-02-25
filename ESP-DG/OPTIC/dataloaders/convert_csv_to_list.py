import os
import pandas as pd


def convert_labeled_list(root, csv_list):
    img_list = list()
    label_list = list()
    domain_label_list = list()
    domain_labels = [0, 1, 2]
    for idx, csv_file in enumerate(csv_list):
        data = pd.read_csv(os.path.join(root, csv_file))
        img_list += data['image'].tolist()
        label_list += data['mask'].tolist()

        domain_label_list += [domain_labels[idx]] * len(data['image'].tolist())
    return img_list, label_list, domain_label_list
