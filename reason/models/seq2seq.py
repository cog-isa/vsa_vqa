import torch
import torch.nn as nn
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader

class Memory(Dataset):
    def __init__(self):
        self.states = []
        self.logprobs = []
        self.advantages = []
        self.rewards = []
        self.values = []
    
    def clear_memory(self):
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.values[:]
        del self.advantages[:]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.logprobs[idx], self.rewards[idx], self.values[idx], self.advantages[idx] 


class Seq2seq(nn.Module):
    """Seq2seq model module
    To do: add docstring to methods
    """
    
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.critic = nn.Sequential(
                nn.Linear(self.decoder.hidden_size, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, 1)
        )
        self.value_loss = nn.MSELoss()
        self.memory = Memory()
        self.old_logprobs = None

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

        state_value = self.critic(encoder_outputs[torch.arange(encoder_outputs.shape[0]), input_lengths - 1])

        self.memory.states.append((x, input_lengths))
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

    def ppo_backward(self, optimizer, reward, value, entropy_factor=0.0):
        losses = []
        grad_output = []
        ppo_epochs = 5
        
        memory_loader = DataLoader(self.memory, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for _ in range(ppo_epochs):
            for state, old_logprobs, reward, value, advantage in tqdm(memory_loader):
                optimizer.zero_grad()

                x, input_lengths = state

                x = x.squeeze()
                input_lengths = input_lengths.squeeze()
                old_logprobs = old_logprobs.squeeze()
                reward = reward.squeeze()
                advantage = advantage.squeeze()

                encoder_outputs, encoder_hidden = self.encoder(x, input_lengths)
                output_symbols, output_logprobs = self.decoder.forward_sample(encoder_outputs, encoder_hidden, reinforce_sample=True)

                value = self.critic(encoder_outputs[torch.arange(encoder_outputs.shape[0]), input_lengths - 1])
                reward_tensor = torch.full_like(value, reward)
                critic_loss = 0.5 * self.value_loss(value, reward_tensor)

                for i, symbol in enumerate(output_symbols):
                    ratios = torch.exp(torch.diag(torch.index_select(output_logprobs[i], 1, symbol)) \
                    - torch.diag(torch.index_select(old_logprobs[i], 1, symbol)))
                    
                    surr1 = ratios * advantage
                    surr2 = torch.clamp(ratios, 0.8, 1.2)  * advantage

                    actor_loss = -torch.min(surr1, surr2).mean() 
                    
                    if torch.isnan(actor_loss).any():
                        print('loss nan')
                        print((output_logprobs[i]).sum())
                        exit(0)

                    losses.append(actor_loss)
                    #grad_output.append(None)

                loss = torch.mean(torch.stack(losses)) + critic_loss 

                loss.backward()
                #torch.autograd.backward(losses, grad_output, retain_graph=True)
                optimizer.step()

                losses = []
                grad_output = []

        self.memory.clear_memory()