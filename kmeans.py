import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from moco_cifar10_demo import ModelBase
import torch


def kmeans(data, K):
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=K, n_init=1, max_iter=1, random_state=0)
    model.fit(data)
    labels = model.labels_
    cnt = {k: len(np.where(labels == k)[0]) for k in range(K)}
    frac = {k: cnt[k] / len(data) for k in range(K)}
    return labels, cnt, frac


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])



train = CIFAR10(root='assets/datasets/cifar10', train=True, transform=train_transform, download=True)
train_loader = DataLoader(train, batch_size=128, shuffle=False, num_workers=16, pin_memory=True)


encoder_q = ModelBase(feature_dim=128, arch='resnet18', bn_splits=8)
state = torch.load('cache-2022-08-02-17-37-47-moco/model.pth')
encoder_q.load_state_dict(state)
encoder_q.eval()
encoder_q.requires_grad_(False)
encoder_q.to('cuda')


features = []
for batch in train_loader:
    features.append(encoder_q(batch[0].to('cuda')))

features = torch.cat(features, dim=0)



labels, cnt, frac = kmeans(features.detach().cpu().numpy(), 10)
print(labels)
print(type(labels))


np.save('moco_torchvision_cifar10_train_cluster.npy', labels)
