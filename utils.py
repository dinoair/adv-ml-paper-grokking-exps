import json

import torch


def save_model(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    return f"Saved model to {path}"


def save_dict(d, path):
    json.dump(d, open(path, 'w'), ensure_ascii=False, indent=4)
    return f"Saved dict to {path}"
