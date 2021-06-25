import torch
import torch.nn as nn


class Seq2seq(nn.Module):
    """Seq2seq model module
    To do: add docstring to methods
    """
    
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y, input_lengths=None):
        encoder_outputs, encoder_hidden = self.encoder(x, input_lengths)
        func_decoder_outputs, arg_decoder_outputs, decoder_hidden = self.decoder(y, encoder_outputs, encoder_hidden)
        return func_decoder_outputs, arg_decoder_outputs

    def sample_output(self, x, input_lengths=None):
        encoder_outputs, encoder_hidden = self.encoder(x, input_lengths)
        output_symbols, _ = self.decoder.forward_sample(encoder_outputs, encoder_hidden)

        symbols = torch.stack(output_symbols)

        if len(list(symbols.shape)) == 2:
            return symbols.transpose(0,1)
        else:
            return symbols.unsqueeze(-1).transpose(0,1)

    def reinforce_forward(self, x, input_lengths=None):
        encoder_outputs, encoder_hidden = self.encoder(x, input_lengths)
        self.output_symbols, self.output_logprobs = self.decoder.forward_sample(encoder_outputs, encoder_hidden, reinforce_sample=True)
        
        symbols = torch.stack(self.output_symbols)

        if len(list(symbols.shape)) == 2:
            return symbols.transpose(0,1)
        else:
            return symbols.unsqueeze(-1).transpose(0,1)

    def reinforce_backward(self, reward, entropy_factor=0.0):
        assert self.output_logprobs is not None and self.output_symbols is not None, 'must call reinforce_forward first'
        losses = []
        grad_output = []
        for i, symbol in enumerate(self.output_symbols):
            if len(self.output_symbols[0].shape) == 1:
                loss = - torch.diag(torch.index_select(self.output_logprobs[i], 1, symbol)).sum()*reward \
                       + entropy_factor*(self.output_logprobs[i]*torch.exp(self.output_logprobs[i])).sum()
                
                if torch.isnan(loss).any():
                    print('loss nan')
                    print((self.output_logprobs[i]).sum())
                    exit(0)
            else:
                loss = - self.output_logprobs[i]*reward

            losses.append(loss.sum())
            grad_output.append(None)
        torch.autograd.backward(losses, grad_output, retain_graph=True)