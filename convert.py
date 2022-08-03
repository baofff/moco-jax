import torch
from collections import OrderedDict


state = torch.load('model.bak.pth')

new_state = OrderedDict()

for key, val in state.items():
    if key.startswith('encoder_q'):
        new_key = key[10:]
        new_state[new_key] = val


print(new_state.keys())

torch.save(new_state, 'model.pth')

