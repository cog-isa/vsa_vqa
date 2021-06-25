import torch 

checkpoint = torch.load('../data/reason/outputs/model_pretrain_ppo/checkpoint.pt')
del checkpoint['net_state']['ppo']
checkpoint['net_params']['ppo'] = 1

torch.save(checkpoint, '../data/reason/outputs/model_pretrain_ppo/checkpoint.pt')