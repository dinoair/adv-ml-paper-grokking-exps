import json
import re

import torch


def save_model(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    return f"Saved model to {path}"


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
