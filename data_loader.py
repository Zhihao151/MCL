from torch import nn
from torchvision import transforms, datasets
from torchvision.transforms import functional as F
from torch.utils.data import random_split, DataLoader, Dataset
import copy
import torch
import numpy as np
import time
import random
import cv2
from PIL import Image
from torchvision.datasets import DatasetFolder, ImageFolder
from tqdm import tqdm
from select_dataset import SelectDataset


def get_train_loader(opt, target_transform=None):
    print('==> Preparing train data..')

    if (opt.dataset == 'CIFAR10'):
        tf_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        trainset = datasets.CIFAR10(root='data/CIFAR10', train=True, download=True)
    elif (opt.dataset == 'CIFAR100'):
        tf_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        trainset = datasets.CIFAR100(root='../../NAD/data/CIFAR100', train=True, download=True)
    elif opt.dataset == 'imagenet':
        tf_train = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        trainset = ImageFolder('../../NAD/data/Imagenet20/train', transform=transforms.Resize((224, 224)))
    elif opt.dataset == 'tinyImagenet':
        tf_train = transforms.Compose([
            # transforms.RandomResizedCrop(64),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        trainset = ImageFolder('../../NAD/data/tiny-imagenet-200/train', transform=transforms.Resize((64, 64)))
    elif opt.dataset == 'gtsrb':

        tmp = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((32, 32))
        ])

        tf_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        trainset = DatasetFolder(
            root='../../NAD/data/GTSRB/train',  # please replace this with path to your training set
            loader=cv2.imread,
            extensions=('ppm',),
            transform=tmp,
            target_transform=None)
    else:
        raise Exception('Invalid dataset')

    train_data = DatasetCL(opt, full_dataset=trainset, transform=tf_train, target_transform=target_transform)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, drop_last=True)

    return train_loader


def get_test_loader(opt):
    print('==> Preparing test data..')
    if (opt.dataset == 'CIFAR10'):
        tf_test = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        testset = datasets.CIFAR10(root='data/CIFAR10', train=False, download=True)
    elif (opt.dataset == 'CIFAR100'):
        tf_test = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        testset = datasets.CIFAR100(root='data/CIFAR100', train=False, download=True)
    elif opt.dataset == 'tinyImagenet':
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])
        testset = ImageFolder('data/tiny-imagenet-200/val', transform=transforms.Resize((64, 64)))
    elif opt.dataset == 'gtsrb':

        tmp = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((32, 32))
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
        ])
        testset = DatasetFolder(
            root='data/GTSRB/test',  # please replace this with path to your test set
            loader=cv2.imread,
            extensions=('ppm',),
            transform=tmp,
            target_transform=None)
    else:
        raise Exception('Invalid dataset')

    test_data_clean = DatasetBD(opt, full_dataset=testset, inject_portion=0, transform=tf_test, mode='test')
    test_data_bad = DatasetBD(opt, full_dataset=testset, inject_portion=1, transform=tf_test, mode='test')

    # (apart from label 0) bad test data
    test_clean_loader = DataLoader(dataset=test_data_clean,
                                   batch_size=opt.batch_size,
                                   shuffle=False,
                                   )
    # all clean test data
    test_bad_loader = DataLoader(dataset=test_data_bad,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 )

    return test_clean_loader, test_bad_loader


def get_backdoor_loader(opt, shuffle=True, batch_size=1):
    print('==> Preparing train data..')
    if (opt.dataset == 'CIFAR10'):
        tf_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        trainset = datasets.CIFAR10(root='data/CIFAR10', train=True, download=True)
    elif opt.dataset == 'gtsrb':

        tmp = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((32, 32))
        ])

        tf_train = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset = DatasetFolder(
            root='./data/GTSRB/train',  # please replace this with path to your training set
            loader=cv2.imread,
            extensions=('ppm',),
            transform=tmp,
            target_transform=None)
    else:
        raise Exception('Invalid dataset')

    train_data_bad = DatasetBD(opt, full_dataset=trainset, inject_portion=opt.inject_portion, transform=tf_train,
                               mode='train')
    train_clean_loader = DataLoader(dataset=train_data_bad,
                                    batch_size=opt.batch_size,
                                    shuffle=shuffle, drop_last=True)

    return train_clean_loader


class DatasetCL(Dataset):
    def __init__(self, opt, full_dataset=None, transform=None, target_transform=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=opt.ratio)
        self.transform = transform
        self.target_transform = target_transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return self.dataLen

    def random_split(self, full_dataset, ratio):
        print('full_train:', len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
        print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))

        return train_dataset


class DatasetBD(Dataset):
    def __init__(self, opt, full_dataset, inject_portion, transform=None, mode="train", device=torch.device("cuda"),
                 distance=1):
        self.dataset = self.addTrigger(full_dataset, opt.target_label, inject_portion, mode, distance, opt.trig_w,
                                       opt.trig_h, opt.trigger_type, opt.target_type, opt.dataset)
        self.device = device
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]

        img = Image.fromarray(img)
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, target_label, inject_portion, mode, distance, trig_w, trig_h, trigger_type,
                   target_type, t_dataset=None):
        print("Generating " + mode + "bad Imgs")
        if target_type != 'cleanLabel' or inject_portion == 1:
            self.perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        else:
            self.perm = []

        # dataset
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]

            if target_type == 'all2one':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]

                    if i in self.perm:
                        # select trigger
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type, t_dataset,
                                                 idx=i)

                        # change target
                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in self.perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type, t_dataset,
                                                 idx=i)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # all2all attack
            elif target_type == 'all2all':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in self.perm:

                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)
                        target_ = self._change_label_next(data[1])

                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in self.perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        target_ = self._change_label_next(data[1])
                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # clean label attack
            elif target_type == 'cleanLabel':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if random.random() < inject_portion * 10:
                        # if i in perm:
                        if data[1] == target_label:
                            self.perm.append(i)
                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type,
                                                     t_dataset, idx=i)

                            dataset_.append((img, data[1]))
                            cnt += 1

                        else:
                            dataset_.append((img, data[1]))
                    else:
                        dataset_.append((img, data[1]))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in self.perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

        time.sleep(0.01)
        print("Injecting Over: " + str(cnt) + "Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")

        return dataset_

    def _change_label_next(self, label):
        label_new = ((label + 1) % 10)
        return label_new

    def selectTrigger(self, img, width, height, distance, trig_w, trig_h, triggerType, t_dataset=None, idx=0):

        assert triggerType in ['squareTrigger', 'gridTrigger', 'randomPixelTrigger', 'fourCornerTrigger',
                               'signalTrigger', 'trojanTrigger', 'blendTrigger', 'centerTrigger', 'wanetTrigger']

        if triggerType == 'squareTrigger':
            img = self._squareTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'gridTrigger':
            img = self._gridTriger(img, width, height, distance, trig_w, trig_h, t_dataset)
        elif triggerType == 'fourCornerTrigger':
            img = self._fourCornerTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'centerTrigger':
            img = self._centerTriger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'blendTrigger':
            img = self._blendTrigger(img, width, height, distance, trig_w, trig_h, t_dataset)
        elif triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'trojanTrigger':
            img = self._trojanTrigger(img, width, height, distance, trig_w, trig_h, t_dataset)
        elif triggerType == 'randomPixelTrigger':
            img = self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'wanetTrigger':
            img = img
        else:
            raise NotImplementedError

        return img

    def _squareTrigger(self, img, width, height, distance, trig_w, trig_h):

        for j in range(width - distance - trig_w, width - distance):
            for k in range(height - distance - trig_h, height - distance):
                img[j, k] = 255.0

        return img

    def _gridTriger(self, img, width, height, distance, trig_w, trig_h, t_dataset):


        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        return img

    def _fourCornerTrigger(self, img, width, height, distance, trig_w, trig_h):
        # right bottom
        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # left top
        img[1][1] = 255
        img[1][2] = 0
        img[1][3] = 255

        img[2][1] = 0
        img[2][2] = 255
        img[2][3] = 0

        img[3][1] = 255
        img[3][2] = 0
        img[3][3] = 0

        # right top
        img[width - 1][1] = 255
        img[width - 1][2] = 0
        img[width - 1][3] = 255

        img[width - 2][1] = 0
        img[width - 2][2] = 255
        img[width - 2][3] = 0

        img[width - 3][1] = 255
        img[width - 3][2] = 0
        img[width - 3][3] = 0

        # left bottom
        img[1][height - 1] = 255
        img[2][height - 1] = 0
        img[3][height - 1] = 255

        img[1][height - 2] = 0
        img[2][height - 2] = 255
        img[3][height - 2] = 0

        img[1][height - 3] = 255
        img[2][height - 3] = 0
        img[3][height - 3] = 0

        return img

    def _centerTriger(self, img, width, height, distance, trig_w, trig_h):

        # adptive center trigger
        alpha = 1
        img[width - 14][height - 14] = 255 * alpha
        img[width - 14][height - 13] = 0 * alpha
        img[width - 14][height - 12] = 255 * alpha

        img[width - 13][height - 14] = 0 * alpha
        img[width - 13][height - 13] = 255 * alpha
        img[width - 13][height - 12] = 0 * alpha

        img[width - 12][height - 14] = 255 * alpha
        img[width - 12][height - 13] = 0 * alpha
        img[width - 12][height - 12] = 0 * alpha

        return img

    def _randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
        blend_img = (1 - alpha) * img + alpha * mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        # print(blend_img.dtype)
        return blend_img

    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        # load signal mask
        signal_mask = np.load('trigger/signal_cifar10_mask.npy')
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        return blend_img

    def _trojanTrigger(self, img, width, height, distance, trig_w, trig_h, t_dataset):
        if t_dataset == 'gtsrb':
            trigger_ptn = Image.open('trigger/trigger_gtsrb.png').convert("RGB")
            trigger_ptn = np.array(trigger_ptn)
            trigger_loc = np.nonzero(trigger_ptn)
            img[trigger_loc] = 0
            img_ = img + trigger_ptn
        elif t_dataset == 'CIFAR10':
            # load trojanmask
            trg = np.load('trigger/best_square_trigger_cifar10.npz')['x']
            # trg.shape: (3, 32, 32)
            trg = np.transpose(trg, (1, 2, 0))
            img_ = np.clip((img + trg).astype('uint8'), 0, 255)
        elif t_dataset == 'CIFAR100':
            trigger_ptn = Image.open('trigger/trigger_cifar100.png').convert("RGB")
            trigger_ptn = np.array(trigger_ptn)
            trigger_loc = np.nonzero(trigger_ptn)
            img[trigger_loc] = 0
            img_ = img + trigger_ptn

        return img_

    def _blendTrigger(self, img, width, height, distance, trig_w, trig_h, t_dataset):
        alpha = 0.2
        poison_img = copy.deepcopy(img)

        # adptive center trigger
        poison_img[width - 14][height - 14] = 255 * alpha + (1 - alpha) * poison_img[width - 14][height - 14]
        poison_img[width - 14][height - 13] = 128 * alpha + (1 - alpha) * poison_img[width - 14][height - 13]
        poison_img[width - 14][height - 12] = 255 * alpha + (1 - alpha) * poison_img[width - 14][height - 12]

        poison_img[width - 13][height - 14] = 128 * alpha + (1 - alpha) * poison_img[width - 13][height - 14]
        poison_img[width - 13][height - 13] = 255 * alpha + (1 - alpha) * poison_img[width - 13][height - 13]
        poison_img[width - 13][height - 12] = 128 * alpha + (1 - alpha) * poison_img[width - 13][height - 12]

        poison_img[width - 12][height - 14] = 255 * alpha + (1 - alpha) * poison_img[width - 12][height - 14]
        poison_img[width - 12][height - 13] = 128 * alpha + (1 - alpha) * poison_img[width - 12][height - 13]
        poison_img[width - 12][height - 12] = 128 * alpha + (1 - alpha) * poison_img[width - 12][height - 12]

        return np.array(poison_img)
