import json
import os
import re

import torch


def save_model(model, optimizer_list, dir_path, filename):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    model_save_path = os.path.join(dir_path, filename)
    save_dict = {'model_state_dict': model.state_dict()}
    for idx, opt in enumerate(optimizer_list):
        save_dict[f"optimizer_{idx}"] = opt.state_dict()
    torch.save(save_dict, model_save_path)
    return f"Saved model to {model_save_path}"


def save_dict(d, path):
    json.dump(d, open(path, 'w', encoding="utf-8"), ensure_ascii=False, indent=4)
    return f"Saved dict to {path}"


def get_triplet_from_sparql(sparql_query):
    triplet = re.findall(r"{(.*?)}", sparql_query)
    if triplet:
        triplet = triplet[0].split()
        triplet = ' '.join([elem for elem in triplet if elem]).strip()
    else:
        triplet = ''
    return triplet


class TXTLogger:
    def __init__(self, work_dir):
        self.save_dir = work_dir
        self.filename = "progress_log.txt"

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.log_file_path = os.path.join(self.save_dir, self.filename)
        log_file = open(self.log_file_path, 'w')
        log_file.close()

    def log(self, data):
        with open(self.log_file_path, 'a') as f:
            f.write(f'{str(data)}\n')
