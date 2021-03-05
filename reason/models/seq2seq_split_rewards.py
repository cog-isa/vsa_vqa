import torch
import torch.nn as nn
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader

class Memory(Dataset):
    def __init__(self):
        self.states = []
        self.state_lengths = []
        self.logprobs = []
        self.advantages = []
        self.rewards = []
        self.values = []

    def cat(self):
        self.states = torch.cat(self.states, dim=0)
        self.state_lengths = torch.cat(self.states, dim=0)
        self.logprobs = torch.cat(self.logprobs, dim=0)
        self.advantages = torch.cat(self.advantages, dim=0)
        self.rewards = torch.cat(self.rewards, dim=0)
        self.values = torch.cat(self.values, dim=0)

    def clear_memory(self):
        del self.states[:]
        del self.state_lengths[:]
        del self.logprobs[:]
        del self.advantages[:]
        del self.rewards[:]
        del self.values[:]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], 
        self.state_lengths[idx],
        self.logprobs[idx], 
        self.advantages[idx], 
        self.rewards[idx], 
        self.values[idx], 


class Seq2seq(nn.Module):
    """Seq2seq model module
    To do: add docstring to methods
    """
    
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.critic = nn.Sequential(
                nn.Linear(self.decoder.hidden_size, 100),
                nn.Tanh(),
                nn.Linear(100, 100),
                nn.Tanh(),
                nn.Linear(100, 1)
        )
        self.value_loss = nn.MSELoss()
        self.memory = Memory()

    def forward(self, x, y, input_lengths=None):
        encoder_outputs, encoder_hidden = self.encoder(x, input_lengths)
        decoder_outputs, decoder_hidden = self.decoder(y, encoder_outputs, encoder_hidden)
        return decoder_outputs

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

    def ppo_forward(self, x, input_lengths=None):
        encoder_outputs, encoder_hidden = self.encoder(x, input_lengths)
        output_symbols, output_logprobs = self.decoder.forward_sample(encoder_outputs, encoder_hidden, reinforce_sample=True)

        state_value = self.critic(encoder_outputs[:, -1, :])

        self.memory.states.append(x)
        self.memory.state_lengths.append(input_lengths)
        self.memory.logprobs.append(torch.stack(output_logprobs).detach())

        return torch.stack(output_symbols).transpose(0,1), state_value.squeeze(-1)

    def reinforce_backward(self, reward, entropy_factor=0.0):
        assert self.output_logprobs is not None and self.output_symbols is not None, 'must call reinforce_forward first'
        losses = []
        grad_output = []
        for i, symbol in enumerate(self.output_symbols):
            if len(self.output_symbols[0].shape) == 1:
                loss = - torch.diag(torch.index_select(self.output_logprobs[i], 1, symbol)).sum()*reward \
                       + entropy_factor*(self.output_logprobs[i]*torch.exp(self.output_logprobs[i])).sum()
            else:
                loss = - self.output_logprobs[i]*reward
                print('im here')
            losses.append(loss.sum())
            grad_output.append(None)
        torch.autograd.backward(losses, grad_output, retain_graph=True)

    def ppo_backward(self, optimizer, reward, value, batch_size=64, entropy_factor=0.0):
        losses = []
        grad_output = []
        ppo_epochs = 4
        
        self.memory.cat()
        memory_loader = DataLoader(self.memory, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

        for _ in range(ppo_epochs):
            for x, input_lengths, old_logprobs, advantage, reward, value in tqdm(memory_loader):
                optimizer.zero_grad()

                state_value = self.critic(encoder_outputs[:, -1, :])

                encoder_outputs, encoder_hidden = self.encoder(x, input_lengths)
                output_symbols, output_logprobs = self.decoder.forward_sample(encoder_outputs, encoder_hidden, reinforce_sample=True)

                for i, symbol in enumerate(output_symbols):
                    if len(output_symbols[0].shape) == 1:
                        ratios = torch.exp(torch.diag(torch.index_select(output_logprobs[i], 1, symbol)) \
                        - torch.diag(torch.index_select(old_logprobs[i], 1, symbol)))
                        
                        surr1 = ratios * advantage
                        surr2 = torch.clamp(ratios, 0.8, 1.2)  * advantage

                        reward_tensor = torch.full_like(state_value, reward)

                        loss = -torch.min(surr1, surr2).mean() + 0.5 * self.value_loss(state_value, reward_tensor)

                        if torch.isnan(loss).any():
                            print('loss nan')
                            print((output_logprobs[i]).sum())
                            exit(0)

                    losses.append(loss)
                    grad_output.append(None)

                torch.autograd.backward(losses, grad_output, retain_graph=True)
                optimizer.step()

                losses = []
                grad_output = []

        self.memory.clear_memory()