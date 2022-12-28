import torch
import torch.nn as nn
import torchvision.models


class Normalize(torch.nn.Module):
    """
    Normalization module for debiasing and normalizing input data values in range [0, 1] with commonly used
    normalization values.
    """
    def __init__(self):
        super(Normalize, self).__init__()
        self.mean = torch.nn.Parameter(torch.as_tensor([0.1307])[None, :, None, None], requires_grad=False)
        self.std = torch.nn.Parameter(torch.as_tensor([0.3081])[None, :, None, None], requires_grad=False)

    def forward(self, x):
        x = x - self.mean
        x = x / self.std
        return x


class MultiAttentionMIL(nn.Module):
    def __init__(self, num_classes: int = 2, num_attentions: int = 3, fc_layer_size: int = 512):
        super().__init__()
        self.num_classes = num_classes
        self.num_attentions = num_attentions

        self.__num_features = 512
        self.fc_layer_size = fc_layer_size
        self.__D = 128

        self.model = torchvision.models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = torch.nn.Sequential(torch.nn.Linear(self.__num_features, self.fc_layer_size))

        normalization = Normalize()
        self.model = torch.nn.Sequential(normalization, self.model)

        self.fc_first = nn.Sequential(
            nn.Linear(self.fc_layer_size, self.__D),
            nn.ReLU(),
        )
        self.attention = nn.Sequential(
            nn.Linear(self.__D, self.__D), nn.Tanh(), nn.Linear(self.__D, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.__D, self.__D),
            nn.ReLU(),
        )

        self.fc_result = nn.Sequential(nn.Linear(self.__D, self.num_classes),
                                       nn.Sigmoid())

    def forward(self, images):
        feature = torch.stack([self.model(__batch) for __batch in images])
        feature = feature.squeeze(-1).squeeze(-1)
        feature = self.fc_first(feature)

        attention = self.attention(feature)
        attention = nn.Softmax(dim=1)(attention)
        attention = torch.transpose(attention, 2, 1)

        m = torch.matmul(attention, feature)
        m = m.view(-1, 1 * self.__D)
        m_tot = m

        for _ in range(self.num_attentions):
            feature = self.fc(feature)

            attention = self.attention(feature)
            attention = nn.Softmax(dim=1)(attention)
            attention = torch.transpose(attention, 2, 1)

            m = torch.matmul(attention, feature)
            m = m.view(-1, 1 * self.__D)
            m_tot += m
        result = self.fc_result(m_tot)

        return result, attention


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = torch.nn.functional.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, A