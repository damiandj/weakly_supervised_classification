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


class Attention(nn.Module):
    """
    Attention module
    """
    def __init__(self, _dim: int = 128):
        super().__init__()
        self._D = _dim
        self.attention = nn.Sequential(
            nn.Linear(self._D, self._D), nn.Tanh(), nn.Linear(self._D, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(self._D, self._D),
            nn.ReLU(),
        )

    def forward(self, _input):
        feature = self.fc(_input)

        attention = self.attention(feature)
        attention = nn.Softmax(dim=1)(attention)
        attention = torch.transpose(attention, 2, 1)

        m = torch.matmul(attention, feature)
        m = m.view(-1, 1 * self._D)

        return feature, attention, m


class FirstAttention(Attention):
    """
    First attention module
    """
    def __init__(self, fc_layer_size: int = 512, _dim: int = 128):
        super().__init__(_dim=_dim)
        self.fc_layer_size = fc_layer_size

        self.fc = nn.Sequential(
            nn.Linear(self.fc_layer_size, self._D),
            nn.ReLU(),
        )


class MultiAttentionMIL(nn.Module):
    """
    Multi Attention Multiple Instance Learning module
    """
    def __init__(self, num_classes: int = 2, num_attentions: int = 3, fc_layer_size: int = 512):
        super().__init__()
        self.num_classes = num_classes
        self.num_attentions = num_attentions

        self.__num_features = 512
        self.fc_layer_size = fc_layer_size
        self._D = 128

        self.model = torchvision.models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = torch.nn.Sequential(torch.nn.Linear(self.__num_features, self.fc_layer_size))

        self.normalization = Normalize()
        self.first_attention = FirstAttention(fc_layer_size=fc_layer_size, _dim=self._D)
        self.attention = Attention(_dim=self._D)

        self.fc_result = nn.Sequential(nn.Linear(self._D, self.num_classes),
                                       nn.Sigmoid())

    def forward(self, images):
        images = self.normalization(images)
        feature = torch.stack([self.model(__batch) for __batch in images])
        feature = feature.squeeze(-1).squeeze(-1)

        feature, attention, m = self.first_attention(feature)
        m_tot = m

        for num in range(self.num_attentions):
            feature, attention, m = self.attention(feature)
            m_tot += m
        result = self.fc_result(m_tot)

        return result, attention
