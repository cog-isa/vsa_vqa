import torch
import torch.nn as nn
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class Memory(Dataset):
    def __init__(self):
        self.states = []
        self.state_lengths = []
        self.output_lengths = []
        self.logprobs = []
        self.advantages = []
        self.rewards = []
        self.values = []

    def cat(self):
        self.states = torch.cat(self.states, dim=0)
        self.state_lengths = torch.cat(self.state_lengths, dim=0)
        self.output_lengths = torch.cat(self.output_lengths, dim=0)
        self.logprobs = torch.cat(self.logprobs, dim=1)
        self.logprobs = self.logprobs.permute(1, 0, 2)
        self.advantages = torch.cat(self.advantages, dim=0)
        self.rewards = torch.cat(self.rewards, dim=0)
        self.values = torch.cat(self.values, dim=0)

    def clear_memory(self):
        del self.states
        del self.state_lengths
        del self.output_lengths
        del self.logprobs
        del self.advantages
        del self.rewards
        del self.values

        self.states = []
        self.state_lengths = []
        self.output_lengths = []
        self.logprobs = []
        self.advantages = []
        self.rewards = []
        self.values = []

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], \
        self.state_lengths[idx], \
        self.output_lengths[idx], \
        self.logprobs[idx], \
        self.advantages[idx], \
        self.rewards[idx], \
        self.values[idx] 

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(512, 256, batch_first=True)
        self.act = nn.Tanh()
        self.clf = nn.Linear(256, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.act(out)
        out = self.clf(out)

        return out

class Seq2seq(nn.Module):
    """Seq2seq model module
    To do: add docstring to methods
    """
    
    def __init__(self, encoder, decoder, ppo=False):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        #self.critic = Critic()
        self.max_grad_norm = 0.1
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
        output_symbols, output_logprobs, decoder_outputs = self.decoder.forward_sample(encoder_outputs, encoder_hidden, 
                                                                    reinforce_sample=True, return_decoder_outputs=True)

        state_value = self.critic(decoder_outputs)

        self.memory.states.append(x)
        self.memory.state_lengths.append(torch.from_numpy(input_lengths))
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

    def ppo_backward(self, optimizer, critic_optimizer, reward, value, batch_size=64, entropy_factor=0.0):
        losses = []
        grad_output = []
        ppo_epochs = 4
        
        self.memory.cat() 
        memory_loader = DataLoader(self.memory, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        
        actor_loss_epoch = 0
        critic_loss_epoch = 0
        advantage_epoch = 0

        for _ in range(ppo_epochs):
            for i, (x, input_lengths, output_lengths, old_logprobs, advantage, reward, value) in enumerate(memory_loader):
                optimizer.zero_grad()
                critic_optimizer.zero_grad()

                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

                encoder_outputs, encoder_hidden = self.encoder(x, input_lengths)
                output_symbols, output_logprobs, decoder_outputs = self.decoder.forward_sample(encoder_outputs, encoder_hidden, 
                                                                    reinforce_sample=True, return_decoder_outputs=True)

                value = self.critic(decoder_outputs.detach()).squeeze()
                mask = torch.arange(value.shape[-1])[None, :] < output_lengths[:, None]
                mask = mask.cuda()
                
                critic_loss = 0.5 * self.value_loss(mask * value, mask * reward)
                critic_loss.backward()
                critic_optimizer.step()
                
                mask = torch.arange(advantage.shape[-1])[None, :] < output_lengths[:, None]
                mask = mask.cuda()
                advantage = advantage * mask

                for i, symbol in enumerate(output_symbols):
                    cur_adv = advantage[:, i]
                    ratios = torch.exp(torch.diag(torch.index_select(output_logprobs[i], 1, symbol)) \
                    - torch.diag(torch.index_select(old_logprobs[:, i, :], 1, symbol)))

                    surr1 = ratios * cur_adv
                    surr2 = torch.clamp(ratios, 0.8, 1.2) * cur_adv

                    actor_loss = -torch.min(surr1, surr2).mean()
                    #actor_loss = - (torch.diag(torch.index_select(output_logprobs[i], 1, symbol)) * cur_adv).sum()
                
                    if torch.isnan(actor_loss).any():
                        print('loss nan')
                        print((output_logprobs[i]).sum())
                        exit(0)

                    losses.append(actor_loss)
                    grad_output.append(None)

                actor_loss = torch.mean(torch.stack(losses))
                #loss = actor_loss + critic_loss 

                #loss.backward()

                if losses[0].requires_grad:
                    torch.autograd.backward(losses, grad_output, retain_graph=True)

                #nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                #nn.utils.clip_grad_norm_(self.decoder.parameters(), self.max_grad_norm)

                optimizer.step()

                actor_loss_epoch += actor_loss.item()
                critic_loss_epoch += critic_loss.item()
                advantage_epoch += advantage.sum().item()

                losses = []
                grad_output = []

        actor_loss_epoch /= ppo_epochs * len(memory_loader) * batch_size
        critic_loss_epoch /= ppo_epochs * len(memory_loader) * batch_size
        advantage_epoch /= ppo_epochs * len(memory_loader) * batch_size

        self.memory.clear_memory()

        return actor_loss_epoch, critic_loss_epoch, advantage_epoch