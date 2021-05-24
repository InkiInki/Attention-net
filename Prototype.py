"""
Author: Inki
Email: inki.yinji@qq.com
Create: 2021 0224
Last modify: 2021 0511
"""

import warnings
import numpy as np
import os as os
from FunctionTool import load_file
warnings.filterwarnings("ignore")


class MIL:

    """
    The father class of MIL.
    :param
        filename:   The given MIL dataset path.
        save_home:  The default temp data save path.
    """
    def __init__(self, filename, save_home="D:/Data/MIL/", bag_space=None):
        self.filename = filename
        self.save_home = save_home
        self.bag_space = bag_space
        self.__init_mil()

    def __init_mil(self):
        """
        The initialize of this class.
        """
        if self.bag_space is None:
            self.bag_space = load_file(self.filename)
        self.num_bag = len(self.bag_space)

        self.bag_size = np.zeros(self.num_bag, dtype=int)
        self.bag_lab = np.zeros_like(self.bag_size, dtype=int)
        self.bag_idx = np.arange(self.num_bag)
        for i in range(self.num_bag):
            self.bag_size[i] = len(self.bag_space[i][0])
            self.bag_lab[i] = self.bag_space[i][1]

        self.num_ins = sum(self.bag_size)
        self.num_att = len(self.bag_space[0, 0][0]) - 1
        self.class_space = list(set(self.bag_lab))
        self.num_class = len(self.class_space)

        self.ins_space = np.zeros((self.num_ins, self.num_att))
        self.ins_idx = np.zeros(self.num_bag + 1, dtype=int)
        self.ins_lab = np.zeros(self.num_ins)
        self.ins_bag_idx = np.zeros(self.num_ins, dtype=int)
        for i in range(self.num_bag):
            self.ins_idx[i + 1] = self.bag_size[i] + self.ins_idx[i]
            self.ins_space[self.ins_idx[i]: self.ins_idx[i + 1]] = self.bag_space[i, 0][:, :self.num_att]
            self.ins_lab[self.ins_idx[i]: self.ins_idx[i + 1]] = self.bag_space[i, 0][:, -1]
            self.ins_bag_idx[self.ins_idx[i]: self.ins_idx[i + 1]] = np.ones(self.bag_size[i]) * i

        self.data_name = self.filename.strip().split("/")[-1].split(".")[0]
        self.zero_ratio = len(self.ins_space[self.ins_space == 0]) / (self.num_ins * self.num_att)
        self.__generate_save_home()

    def __generate_save_home(self):
        """
        Generate the save home.
        """
        if not os.path.exists(self.save_home):
            os.makedirs(self.save_home)

    def get_data_info(self):
        """
        Print the data set information.
        """
        temp_idx = 5 if self.num_bag > 5 else self.num_bag
        print("The {}'s information is:".format(self.data_name), "\n"
              "Number bags:", self.num_bag, "\n"
              "Class space:", self.class_space, "\n"
              "Number classes:", self.num_class, "\n"
              "Bag size:", self.bag_size[:temp_idx], "...\n"
              "Bag label", self.bag_lab, "...\n"
              "Maximum bag's size:", np.max(self.bag_size), "\n"
              "Minimum bag's size:", np.min(self.bag_size), "\n"
              "Zero ratio:", self.zero_ratio, "\n"
              "Number instances:", self.num_ins, "\n"
              "Instance dimensions:", self.num_att, "\n"
              "Instance index:", self.ins_idx[: temp_idx], "...\n"
              "Instance label:", self.ins_lab[: temp_idx], "...\n"
              "Instance label corresponding bag'S index:", self.ins_bag_idx[:temp_idx], "...\n")

    def get_sub_ins_space(self, bag_idx):
        """
        Given a bag idx array, and return a subset of instance space.
        """
        num_ins = sum(self.bag_size[bag_idx])
        ret_ins_space = np.zeros((num_ins, self.num_att))
        ret_ins_label = np.zeros(num_ins)
        ret_ins_bag_idx = np.zeros(num_ins, dtype=int)
        count = 0
        for i in bag_idx:
            bag_size = self.bag_size[i]
            ret_ins_space[count: count + bag_size] = self.bag_space[i, 0][:, :-1]
            ret_ins_label[count: count + bag_size] = self.bag_lab[i]
            ret_ins_bag_idx[count: count + bag_size] = i
            count += bag_size

        return ret_ins_space, ret_ins_label, ret_ins_bag_idx


if __name__ == '__main__':
    from MnistLoadTool import MnistLoader
    mnist = MnistLoader(seed=1, mnist_path="D:/Data/OneDrive/文档/Code/MIL1/Data")
    mil = MIL("mnist.none", bag_space=mnist.bag_space)
    mil.get_data_info()
