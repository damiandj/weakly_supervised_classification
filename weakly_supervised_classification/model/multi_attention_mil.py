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
