Author：Inki
Contact：inki.yinji@qq.com
============================================================
The paper name
    --> 2018 Attention-based Deep Multiple Instance Learning
============================================================
Parameters that can be modified:
1) If you given a .mat form data set：
    The data source of the mat file you can refer this website
        --> https://gitee.com/inkiinki/data20201205/tree/master/Data20201205
    a) file_name:
        --> The data path of the mat file.
    b) epoch:
        --> The recommended parameters
            --> Musk, Elephant, Fox, Tiger: 100
            --> Newsgroups, Web: 5
    c) net_type:
        --> "a": attention-net
        --> "g": gated-attention-net
    d) loops:
        --> The times of k-cv.
    e) others:
        --> Please refer the source code, the origin paper or contact to me.
    Note:
        The convolutional layer has been removed.
============================================================
2) If you want to use the MNIST data set:
    You must modify these code
        an = AttentionNet(file_name, epoch=5, net_type="a")
        ↓↓↓
        an = AttentionNet(file_name, epoch=5, net_type="a", bag_space=bag_space)
    and remove the comment of
        # from MnistLoadTool import MnistLoader
        # bag_space = MnistLoader(seed=1, po_label=po_label, data_type="mnist", data_path=file_name).bag_space
    and the data_type must be "mnist".
    Note:
        The data_path for MnistLoader is the saved path of the MNIST data set.
        If the MNIST data set hasn't been downloaded, it will be automatically downloaded to the default location or the given data_path.
    a) seed:
        --> To make the fairness of the experiments, we recommend you use a fixed number.
    b) po_label:
        The positive bags belong to the po_label-class (c-class), the negative bags are randomly selected from the other-class.
        For MNIST:
            You must set po_label to c \in [0..9]
    Additionally, you can given a .csv data set, the data source you can refer the traditional iris.csv data set.
    And at here, we given a simple describe:
        ↓↓↓
    The data format \in \mathbb{R}^{n \times (d + 1)}, where n is the number of the instances, the d is the dimensionality.
        --> [[instance1, label1],
             [instance2, label2],
             ...
             [instance_n, label_n]]
============================================================
Copyright: None.
Times: 20210524