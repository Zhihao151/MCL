import torch
from torchvision import transforms as T

class Normalizer:
    def __init__(self, opt):
        self.normalizer = self._get_normalizer(opt)

    def _get_normalizer(self, dataset):
        if dataset == "CIFAR10" or dataset == "CIFAR100":
            mean = torch.FloatTensor([0.4914, 0.4822, 0.4465])
            std = torch.FloatTensor([0.2023, 0.1994, 0.2010])
        elif dataset == "imagenet":
            mean = torch.FloatTensor([0.485, 0.456, 0.406])
            std = torch.FloatTensor([0.229, 0.224, 0.225])
        elif dataset == "tinyImagenet":
            mean = torch.FloatTensor([0.5, 0.5, 0.5])
            std = torch.FloatTensor([0.5, 0.5, 0.5])
        elif dataset == "gtsrb" or dataset == "celeba":
            mean = torch.FloatTensor([0.3403, 0.3121, 0.3214])
            std = torch.FloatTensor([0.2724, 0.2608, 0.2669])
        else:
            raise Exception("Invalid dataset")
        normalizer = T.Normalize(mean, std)
        return normalizer

    def __call__(self, x):
        if self.normalizer:
            x = self.normalizer(x)
        return x

class Denormalizer:
    def __init__(self, opt):
        self.denormalizer = self._get_denormalizer(opt)

    def _get_denormalizer(self, dataset):
        if dataset == "CIFAR10" or dataset == "CIFAR100":
            mean = torch.FloatTensor([0.4914, 0.4822, 0.4465])
            std = torch.FloatTensor([0.2023, 0.1994, 0.2010])
        elif dataset == "imagenet":
            mean = torch.FloatTensor([0.485, 0.456, 0.406])
            std = torch.FloatTensor([0.229, 0.224, 0.225])
        elif dataset == "tinyImagenet":
            mean = torch.FloatTensor([0.5, 0.5, 0.5])
            std = torch.FloatTensor([0.5, 0.5, 0.5])
        elif dataset == "gtsrb" or dataset == "celeba":
            mean = torch.FloatTensor([0.3403, 0.3121, 0.3214])
            std = torch.FloatTensor([0.2724, 0.2608, 0.2669])
        else:
            raise Exception("Invalid dataset")

        if mean.__len__() == 1:
            mean = - mean
        else:  # len > 1
            mean = [-i for i in mean]

        if std.__len__() == 1:
            std = 1 / std
        else:  # len > 1
            std = [1 / i for i in std]

        # copy from answer in
        # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
        # user: https://discuss.pytorch.org/u/svd3

        denormalizer = T.Compose([
            T.Normalize(mean=[0., 0., 0.],
                                 std=std),
            T.Normalize(mean=mean,
                                 std=[1., 1., 1.]),
        ])

        return denormalizer

    def __call__(self, x):
        if self.denormalizer:
            x = self.denormalizer(x)
        return x

