import json
import os

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from seq2seq_dataset import Text2SparqlDataset
from seq2seq_predictor import Seq2SeqPredictor
from seq2seq_trainer import Seq2SeqTrainer
from sparql_tokenizer import SPARQLTokenizer


def main():
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    config = yaml.load((open(os.path.join(os.environ['PROJECT_PATH'], "configs/config.yaml"), 'r')), Loader=yaml.Loader)

    tokenizer_name = config['hf_tokenizer']
    RU_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)

    train_data = json.load(open(config['data']['train'], 'r'))
    test_data = json.load(open(config['data']['test'], 'r'))

    train_sparql_list = [sample['masked_sparql_query'] for sample in train_data]
    test_sparql_list = [sample['masked_sparql_query'] for sample in test_data]

    SPARQL_TOKENIZER = SPARQLTokenizer(train_sparql_list, pad_flag=True)

    test_tokenized_sparqls_list = np.array([SPARQL_TOKENIZER(sparql_query) for sparql_query in test_sparql_list])

    test_questions_list = [sample['filled_paraphrased_canonical'] for sample in test_data]

    test_tokenized_questions_list = RU_TOKENIZER(test_questions_list, return_tensors="pt", padding=True, truncation=True)

    test_dataset = Text2SparqlDataset(tokenized_question_list=test_tokenized_questions_list,
                                     tokenized_sparql_list=test_tokenized_sparqls_list,
                                     question_list=test_questions_list,
                                     sparql_list=test_sparql_list,
                                     dev=DEVICE)

    batch_size = config['batch_size']
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    saved_model_path = os.path.join(os.environ['PROJECT_PATH'], config['save_model_path'],
                                    config['inference_model_name'])
    model_checkpoint = torch.load(saved_model_path, map_location=torch.device(DEVICE))
    encoder_state_dict = model_checkpoint['encoder_state_dict']
    decoder_state_dict = model_checkpoint['decoder_state_dict']

    trainer = Seq2SeqTrainer(config=config, device=DEVICE, target_tokenizer=SPARQL_TOKENIZER, train_phase=False)
    trainer.encoder.load_state_dict(encoder_state_dict)
    trainer.decoder.load_state_dict(decoder_state_dict)

    predictor = Seq2SeqPredictor(config=config, criterion=nn.NLLLoss(),
                                 target_tokenizer=SPARQL_TOKENIZER, device=DEVICE)

    prediction_result = predictor.predict(encoder=trainer.encoder, decoder=trainer.decoder,
                                          dataloader=test_dataloader)

    print('Exact match: ', prediction_result["exact_match_score"])


if __name__ == "__main__":
    main()