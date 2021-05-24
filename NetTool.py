"""
Author: Inki
Email: inki.yinji@qq.com
Create: 2021 0406
Last modify: 2021 0510
"""

import torch
import torch.nn as nn
import torch.nn.functional as func_nn
import warnings

warnings.filterwarnings('ignore')


class ANET(nn.Module):

    def __init__(self,
                 num_att: int,
                 num_class: int,
                 mapping_len: int = 100
                 ):
        super(ANET, self).__init__()
        self.L = 512
        self.D = 128
        self.H1 = 128
        self.H2 = mapping_len

        self.feature_extractor = nn.Sequential(
            nn.Linear(num_att, self.L),
            nn.ReLU(),
        )

        self.attention_v = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
        )

        self.attention_u = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.L)

        self.mapping = nn.Sequential(
            nn.Linear(self.L, self.H1),
            nn.Sigmoid(),
            nn.Linear(self.H1, self.H2),
            nn.Sigmoid(),
            nn.Linear(self.H2, num_class),
        )

    def forward(self, x):
        x_f = self.feature_extractor(x)
        x_a_v = self.attention_v(x_f)
        x_a_u = self.attention_u(x_f)

        x_a_w = self.attention_weights(x_a_v * x_a_u)
        x_a_w = func_nn.softmax(x_a_w, dim=1)

        x_m = x_a_w + x_f
        x_o = self.mapping(x_m)
        return x_o


class ENET(nn.Module):

    def __init__(self,
                 num_att: int,
                 num_class: int,
                 mapping_len: int = 100
                 ):
        super(ENET, self).__init__()
        self.H1 = 128
        self.H2 = mapping_len

        self.fc = nn.Sequential(  # The fully connected layer
            nn.Linear(num_att, self.H1),
            nn.Sigmoid(),
            nn.Linear(self.H1, self.H2),
            nn.Sigmoid(),
            nn.Linear(self.H2, num_class)
        )

    def forward(self, x):
        return self.fc(x)


class Attention(nn.Module):
    def __init__(self, num_att, D):
        super(Attention, self).__init__()
        self.num_att = num_att
        self.L = 500
        self.D = D
        self.K = 1

        self.feature_extractor_part = nn.Sequential(
            nn.Linear(self.num_att, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part(x)

        A = self.attention(H)
        A = torch.transpose(A, 1, 0)
        A = func_nn.softmax(A, dim=1)
        M = torch.mm(A, H)

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        pred = 1 if Y_hat.eq(Y).cpu().item() else 0
        error = 1. - pred

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A


class GatedAttention(nn.Module):
    def __init__(self, num_att, D):
        super(GatedAttention, self).__init__()
        self.num_att = num_att
        self.L = 500
        self.D = D
        self.K = 1

        self.feature_extractor_part = nn.Sequential(
            nn.Linear(self.num_att, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part(x)  # NxL

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = func_nn.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A


def bp_mip_loss(x, y):
    x = torch.max(torch.max(x[0], 1).values)
    return torch.mean(torch.pow((x - y), 2))


if __name__ == '__main__':
    torch.manual_seed(1)
    data = torch.rand(1, 5, 10)
    label = torch.tensor([1])
    net = BPNet(10, 10)
    net.calculate_classification_error(data, label)
