import torch

def train_on_batch(input_data, target_data, encoder, decoder,
                   encoder_optimizer, decoder_optimizer, criterion,
                   batch_size, target_max_len, dev):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_output = encoder(input_data)
    pooler = encoder_output['pooler']

    decoder_input = torch.tensor([[0] * batch_size], dtype=torch.long, device=dev).view(1, batch_size, 1)
    decoder_hidden = pooler.view(1, batch_size, -1)

    target_tensor = target_data['input_ids'].view(batch_size, target_max_len, 1)
    target_length = target_tensor.shape[1]
    loss = 0
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, batch_size)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.reshape(1, batch_size, 1)
        # на вход функции ошибки - распределение выхода, верные индексы на данном этапе - уже юзаем reduction mean - делить на батч еще не надо
        loss += criterion(decoder_output.squeeze(), target_tensor[:, di, :].squeeze())

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


def evaluate_batch(input_tensor, encoder, decoder,
                   criterion, batch_size,
                   sparql_tokenizer,
                   target_max_len, device,
                   target_data=None):
    result_dict = dict()
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        encoder_output = encoder(input_tensor)
        pooler = encoder_output['pooler']

        decoder_input = torch.tensor([[0] * batch_size], dtype=torch.long, device=device).view(1, batch_size, 1)
        decoder_hidden = pooler.view(1, batch_size, -1)
        loss = 0

        decoder_result_list = []
        for di in range(target_max_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, batch_size)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.reshape(1, batch_size, 1)
            # на вход функции ошибки - распределение выхода, верные индексы на данном этапе.
            if target_data is not None:
                target_tensor = target_data['input_ids'].view(batch_size, target_max_len, 1)
                loss += criterion(decoder_output.squeeze(), target_tensor[:, di, :].squeeze())
            decoder_result_list.append(list(decoder_input.flatten().numpy()))

        if target_data is not None:
            result_dict['loss'] = loss.item()

        # TODO: корявый способ обработки предсказаний для батча
        batch_preds_list = [[] for _ in range(batch_size)]
        for batch in decoder_result_list:
            for idx, sample_idx in enumerate(batch):
                query_token = sparql_tokenizer.index2word[sample_idx]
                batch_preds_list[idx].append(query_token)

        for idx in range(batch_size):
            filtered_sample = list(filter(lambda x: x not in ['SOS', 'EOS', 'PAD'], batch_preds_list[idx]))
            query = " ".join(filtered_sample)
            batch_preds_list[idx] = query

        result_dict['predicted_query'] = batch_preds_list

    return result_dict

