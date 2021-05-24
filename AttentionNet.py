# coding: utf-8
"""
Author: Inki
Email: inki.yinji@qq.com
Create: 2021 0510
Last modify: 2021 0510
"""

import numpy as np
import torch
import torch.optim as opt
import torch.utils.data as data_utils
from sklearn.metrics import accuracy_score, f1_score
from Prototype import MIL
from FunctionTool import get_k_cross_validation_index
from NetTool import GatedAttention, Attention


class BagLoader(data_utils.Dataset):

    def __init__(self, bags, bags_label, idx=None):
        """"""
        self.bags = bags
        self.idx = idx
        if self.idx is None:
            self.idx = list(range(len(bags)))
        self.num_idx = len(self.idx)
        self.bags_label = bags_label[idx]

    def __getitem__(self, idx):
        bag = [self.bags[self.idx[idx], 0][:, :-1].tolist()]
        bag = torch.from_numpy(np.array(bag))

        return bag.double(), torch.tensor([self.bags_label[idx].tolist()]).double()

    def __len__(self):
        """"""
        return self.num_idx


class AttentionNet(MIL):

    def __init__(self, file_name,
                 epoch=10,
                 lr=0.001,
                 net_type="a",
                 bag_space=None):
        """
        :param net_type: "a" or "g"
        """
        super(AttentionNet, self).__init__(file_name, bag_space=bag_space)
        self.epoch = epoch
        self.lr = lr
        self.net_type = net_type
        self.net = None
        self.opt = None
        self.loss = torch.nn.CrossEntropyLoss()

    def __get_optimizer(self):
        self.opt = opt.Adam(self.net.parameters(), self.lr)

    def main(self):
        tr_idxes, te_idxes = get_k_cross_validation_index(self.num_bag)
        pre_label_list, label_list = [], []
        for tr_idx, te_idx in zip(tr_idxes, te_idxes):
            tr_loader = BagLoader(self.bag_space, self.bag_lab, tr_idx)
            te_loader = BagLoader(self.bag_space, self.bag_lab, te_idx)
            if self.net_type == "a":
                self.net = Attention(self.num_att, self.num_bag)
            elif self.net_type == "g":
                self.net = GatedAttention(self.num_att, self.num_bag)
            self.__get_optimizer()
            batch_count = 0
            for epoch in range(self.epoch):
                tr_loss, tr_error = 0, 0
                for data, label in tr_loader:
                    loss, _ = self.net.calculate_objective(data.float(), label)
                    self.opt.zero_grad()
                    tr_loss += loss.data[0]
                    error, _ = self.net.calculate_classification_error(data.float(), label)
                    loss.backward()
                    self.opt.step()
                    tr_error += error
                    batch_count += 1
                # print("Epoch %d, loss %.4f, error %.4f" % (epoch + 1, tr_loss / batch_count,
                #                                                tr_error / batch_count))

            for data, label in te_loader:
                _, pre_lab = self.net.calculate_classification_error(data.float(), label)
                pre_label_list.extend(pre_lab[0].numpy())
            label_list.extend(self.bag_lab[te_idx].tolist())
        acc, f1 = accuracy_score(pre_label_list, label_list), f1_score(pre_label_list, label_list)
        return acc, f1


if __name__ == '__main__':
    import time

    s_t = time.time()
    po_label = 2
    file_name = "D:/Data/OneDrive/文档/Code/MIL1/Data/Text/alt_atheism.mat"
    # from MnistLoadTool import MnistLoader
    # bag_space = MnistLoader(seed=1, po_label=po_label, data_type="mnist", data_path=file_name).bag_space
    an = AttentionNet(file_name, epoch=5, net_type="a")
    print(file_name.split("/")[-1].split(".")[0], "a")

    loops = 5

    acc_list, f1_list = [], []
    for i in range(loops):
        t_acc, t_f1 = an.main()
        acc_list.append(t_acc)
        f1_list.append(t_f1)
        print("%.4f; %.4f" % (t_acc, t_f1))
    print("$%.3lf\\pm%.3lf$; $%.3lf\\pm%.3lf$" % (np.sum(acc_list) / loops, np.std(acc_list),
                                                  np.sum(f1_list) / loops, np.std(f1_list)))
    print("%.4f" % (time.time() - s_t))
