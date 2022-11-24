import torch


def save_seq2seq(encoder, encoder_opt, decoder, decoder_opt, path):
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_encoder_state_dict': encoder_opt.state_dict(),
        'optimizer_decoder_state_dict': decoder_opt.state_dict()
    }, path)
    return f"Saved seq2seq model to {path}"
