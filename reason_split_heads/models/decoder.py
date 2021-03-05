import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .base_rnn import BaseRNN 
from .attention import Attention
import math


class Decoder(BaseRNN):
    """Decoder RNN module
    To do: add docstring to methods
    """
    
    def __init__(self, vocab, vocab_size, max_len, word_vec_dim, hidden_size,
                 n_layers, start_id=1, end_id=2, rnn_cell='lstm',
                 bidirectional=False, input_dropout_p=0,
                 dropout_p=0, use_attention=False):
        super(Decoder, self).__init__(vocab_size, max_len, hidden_size, 
                                      input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.vocab = vocab
        self.func_size = len(vocab['func_token_to_idx'])
        self.arg_size = len(vocab['arg_token_to_idx'])
        self.max_length = max_len
        self.output_size = vocab_size
        self.hidden_size = hidden_size
        self.word_vec_dim = word_vec_dim
        self.bidirectional_encoder = bidirectional
        if bidirectional:
            self.hidden_size *= 2
        self.use_attention = use_attention
        self.start_id = start_id
        self.end_id = end_id

        self.embedding = nn.Embedding(self.output_size, self.word_vec_dim)
        self.rnn = self.rnn_cell(self.word_vec_dim, self.hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.out_linear = nn.Linear(self.hidden_size, self.output_size)
        
        self.out_func = nn.Linear(self.hidden_size, self.func_size)
        self.out_arg = nn.Linear(self.hidden_size, self.arg_size)

        if use_attention:
            self.attention = Attention(self.hidden_size)

        # self.mask = torch.zeros((self.func_size, self.arg_size)).cuda()

        # for idxs in self.vocab['program_idx_couple'].keys():
        #     func_idx, arg_idx = idxs.split(' ')

        #     func_idx = int(func_idx)
        #     arg_idx = int(arg_idx)

        #     self.mask[func_idx, arg_idx] = 1.0

    def forward_step(self, input_var, hidden, encoder_outputs):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)

        embedded = self.input_dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        func_output = self.out_func(output.contiguous().view(-1, self.hidden_size))
        arg_output = self.out_arg(output.contiguous().view(-1, self.hidden_size))


        func_predicted_softmax = F.log_softmax(func_output.view(batch_size, output_size, -1), 2)
        arg_predicted_softmax = F.log_softmax(arg_output.view(batch_size, output_size, -1), 2)
        return func_predicted_softmax, arg_predicted_softmax, hidden, attn

    def forward(self, y, encoder_outputs, encoder_hidden):
        decoder_hidden = self._init_state(encoder_hidden)
        func_decoder_outputs, arg_decoder_outputs, decoder_hidden, attn = self.forward_step(y, decoder_hidden, encoder_outputs)
        return func_decoder_outputs, arg_decoder_outputs, decoder_hidden

    def forward_sample(self, encoder_outputs, encoder_hidden, reinforce_sample=False):
        if isinstance(encoder_hidden, tuple):
            batch_size = encoder_hidden[0].size(1)
            use_cuda = encoder_hidden[0].is_cuda
        else:
            batch_size = encoder_hidden.size(1)
            use_cuda = encoder_hidden.is_cuda
        decoder_hidden = self._init_state(encoder_hidden)    
        decoder_input = Variable(torch.LongTensor(batch_size, 1).fill_(self.start_id))
        if use_cuda:
            decoder_input = decoder_input.cuda()


        output_logprobs = []
        output_symbols = []
        output_lengths = np.array([self.max_length] * batch_size)

        def decode(i, func_output, arg_output, reinforce_sample=reinforce_sample):
            output_prob = torch.zeros(batch_size, len(self.vocab['program_idx_couple'].keys())).cuda()

            prob_matrix = torch.bmm(torch.exp(func_output.view(batch_size, -1, 1)), torch.exp(arg_output.view(batch_size, 1, -1)))
            
            for idxs, value in self.vocab['program_idx_couple'].items():
                func_idx, arg_idx = idxs.split(' ')

                func_idx = int(func_idx)
                arg_idx = int(arg_idx)

                output_prob[:, value] = prob_matrix[:, func_idx, arg_idx]

            output_prob = output_prob / output_prob.sum(dim=1).view(batch_size, 1)

            output_logprob = torch.log(output_prob)
            
            output_logprobs.append(output_logprob)

            # unk_prob = torch.ones(batch_size).cuda()

            # for idxs in self.vocab['program_idx_couple'].keys():
            #     if idxs != '22 4':
            #         value = self.vocab['program_idx_couple'][idxs]

            #         func_idx, arg_idx = idxs.split(' ')
            #         func_idx = int(func_idx)
            #         arg_idx = int(arg_idx)

            #         logprob = func_output[:, :, func_idx] + arg_output[:, :, arg_idx]
            #         logprob = logprob.squeeze()
            #         output_logprob[:, value] = logprob

            #         unk_prob -= torch.exp(logprob)

            # unk_prob[unk_prob < 1e-7] = 1e-7

            

            # xsum = 0.0
            # unk_xsum = 0.0

            # for func_idx in range(func_output.shape[-1]):
            #     for arg_idx in range(arg_output.shape[-1]):
            #         logprob = func_output[0, :, func_idx] + arg_output[0, :, arg_idx]
            #         xsum += torch.exp(logprob).item()

            #         if self.vocab['program_idx_couple'].get('{} {}'.format(func_idx, arg_idx, 3)) == 3:
            #             unk_xsum += torch.exp(logprob).item()

            # print(xsum)
            # print(unk_xsum)
            # exit(0)

            if reinforce_sample:
                dist = torch.distributions.Categorical(probs=output_prob)
                symbols = dist.sample().unsqueeze(1)
                # prob_matrix = torch.bmm(torch.exp(func_output.view(batch_size, -1, 1)), torch.exp(arg_output.view(batch_size, 1, -1))) * self.mask
                
                # joint_dist = torch.distributions.Categorical(probs=prob_matrix.view(batch_size, -1))
                # joint_sample = joint_dist.sample().unsqueeze(1)

                # func_symbols = joint_sample / self.arg_size 
                # arg_symbols = joint_sample % self.arg_size

                #func_dist = torch.distributions.Categorical(probs=torch.exp(func_output.view(batch_size, -1)))
                #arg_dist = torch.distributions.Categorical(probs=torch.exp(arg_output.view(batch_size, -1))) 
                
                #func_symbols = func_dist.sample().unsqueeze(1)
                #arg_symbols = arg_dist.sample().unsqueeze(1)
                
            else:
                symbols = output_prob.topk(1)[1].view(batch_size, -1)

                # applies mask to inputs in order not to predict <unk>, reduces quality 
                # prob_matrix = torch.bmm(torch.exp(func_output.view(batch_size, -1, 1)), torch.exp(arg_output.view(batch_size, 1, -1))) * self.mask
                # equivalent to usual argmax sampling
                # prob_matrix = torch.bmm(torch.exp(func_output.view(batch_size, -1, 1)), torch.exp(arg_output.view(batch_size, 1, -1))) 

                # argmax_vals = (torch.max(prob_matrix.view(batch_size, -1), dim=1)[1]).unsqueeze(1)
                # func_symbols = argmax_vals / self.arg_size 
                # arg_symbols = argmax_vals % self.arg_size

                #func_symbols = func_output.topk(1)[1].view(batch_size, -1)
                #arg_symbols = arg_output.topk(1)[1].view(batch_size, -1)


            # symbols = torch.zeros_like(func_symbols)

            # for i in range(batch_size):
            #     symbols[i, :] = self.vocab['program_idx_couple'].get('{} {}'.format(func_symbols[i, :].item(), arg_symbols[i, :].item()), 3)

            output_symbols.append(symbols.squeeze())

            eos_batches = symbols.data.eq(self.end_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((output_lengths > i) & eos_batches) != 0
                output_lengths[update_idx] = len(output_symbols)

            return symbols

        for i in range(self.max_length):
            func_decoder_output, arg_decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = decode(i, func_decoder_output, arg_decoder_output)

        return output_symbols, output_logprobs

    def _init_state(self, encoder_hidden):
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h