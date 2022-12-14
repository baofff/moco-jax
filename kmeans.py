import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision.models import resnet
from functools import partial
import einops
from PIL import Image
import pickle
from scipy.optimize import linear_sum_assignment


class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training and self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)


class ModelBase(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """

    def __init__(self, feature_dim=128, arch=None, bn_splits=16):
        super(ModelBase, self).__init__()

        # use split batchnorm
        norm_layer = partial(SplitBatchNorm, num_splits=bn_splits) if bn_splits > 1 else nn.BatchNorm2d
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        self.net = []
        for name, module in net.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))
            self.net.append(module)

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.net(x)
        # note: not normalized here
        return x


def kmeans(data, K):
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=K, n_init=100, random_state=0)
    model.fit(data)
    labels = model.labels_
    cnt = {k: len(np.where(labels == k)[0]) for k in range(K)}
    frac = {k: cnt[k] / len(data) for k in range(K)}
    return labels, cnt, frac, model


def save_img_by_cluster(data, labels, K, fname):
    dct = {k: [] for k in range(K)}
    for i, label in enumerate(labels):
        dct[label].append(i)

    imgs = []
    for k in range(K):
        for i in dct[k][:10]:
            imgs.append(data[i])

    imgs = np.array(imgs)
    print(len(imgs))
    imgs = einops.rearrange(imgs, '(ncol nrow) H W C -> (ncol H) (nrow W) C', ncol=K, nrow=10)
    Image.fromarray(imgs).save(f'{fname}.png')


def entropy(p):
    return -sum(p * np.log(p))


def distance(batch, clusters):
    return ( np.sum(batch ** 2, axis=1, keepdims=True)
            + np.sum(clusters ** 2, axis=1)
            - 2 * np.matmul(batch, clusters.T) )


def my_predict(model, features):
    dist = distance(features, model.cluster_centers_)
    pred = np.argmin(dist, axis=1)
    return pred


def calculate_acc(ypred, y, return_idx=False):
    """
    Calculating the clustering accuracy. The predicted result must have the same number of clusters as the ground truth.

    ypred: 1-D numpy vector, predicted labels
    y: 1-D numpy vector, ground truth
    The problem of finding the best permutation to calculate the clustering accuracy is a linear assignment problem.
    This function construct a N-by-N cost matrix, then pass it to scipy.optimize.linear_sum_assignment to solve the assignment problem.

    """
    assert len(y) > 0
    assert len(np.unique(ypred)) == len(np.unique(y))

    s = np.unique(ypred)
    t = np.unique(y)

    N = len(np.unique(ypred))
    C = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            idx = np.logical_and(ypred == s[i], y == t[j])
            C[i][j] = np.count_nonzero(idx)

    # convert the C matrix to the 'true' cost
    Cmax = np.amax(C)
    C = Cmax - C
    row, col = linear_sum_assignment(C)
    # calculating the accuracy according to the optimal assignment
    count = 0
    for i in range(N):
        idx = np.logical_and(ypred == s[row[i]], y == t[col[i]])
        count += np.count_nonzero(idx)

    if return_idx:
        return 1.0 * count / len(y), row, col
    else:
        return 1.0 * count / len(y)


def main(K=50):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


    train = CIFAR10(root='assets/datasets/cifar10', train=True, transform=transform, download=True)
    train_loader = DataLoader(train, batch_size=128, shuffle=False, num_workers=16, pin_memory=True)
    test = CIFAR10(root='assets/datasets/cifar10', train=False, transform=transform, download=True)
    test_loader = DataLoader(test, batch_size=128, shuffle=False, num_workers=16, pin_memory=True)


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

    labels, cnt, frac, model = kmeans(features.detach().cpu().numpy(), K)
    print(frac)
    print(entropy([frac[k] for k in range(K)]), np.log(K))
    if K == 10:
        print('train acc', calculate_acc(labels, train.targets))

    fname = f'moco_torchvision_cifar10_train_cluster_{K}'
    with open(f'{fname}.pkl', 'wb') as f:
        pickle.dump(model, f)

    np.save(f'{fname}.npy', labels)
    save_img_by_cluster(train.data, labels, K, fname)

    features_test = []
    for batch in test_loader:
        features_test.append(encoder_q(batch[0].to('cuda')))

    features_test = torch.cat(features_test, dim=0)
    pred = model.predict(features_test.detach().cpu().numpy())
    my_pred = my_predict(model, features_test.detach().cpu().numpy())
    print(pred)
    print(my_pred)
    assert (pred == my_pred).all()
    print('pred = my_pred')
    if K == 10:
        print('test acc', calculate_acc(pred, test.targets))


    fname_test = f'moco_torchvision_cifar10_test_cluster_{K}'
    np.save(f'{fname_test}.npy', pred)
    save_img_by_cluster(test.data, pred, K, fname_test)


if __name__ == "__main__":
    main(K=50)
