import pylab as p
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),

            nn.Conv1d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(output_channel)
        )

        self.skip_connection = nn.Sequential()
        if output_channel != input_channel:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_channel)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.skip_connection(x) + out
        out = self.relu(out)
        return out


class base_Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        model_output_dim = configs.features_len
        # model_output_dim = 125
        self.layer1 = ResBlock(input_channel=configs.input_channels, output_channel=8, stride=1)
        self.layer2 = ResBlock(input_channel=8, output_channel=16, stride=2)
        self.layer3 = ResBlock(input_channel=16, output_channel=24, stride=2)
        self.layer4 = ResBlock(input_channel=24, output_channel=16, stride=1)
        self.layer5 = ResBlock(input_channel=16, output_channel=configs.final_out_channels, stride=1)
        self.linear = nn.Linear(configs.final_out_channels * model_output_dim, 1)

    def forward(self, x):
        N,L = x.shape[0], x.shape[1]
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out_feature = self.layer5(out)
        logits = self.linear(out_feature.view(N, -1))

        return logits, out_feature

#
# def count_parameters(model):
#     count = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print('The model has {} trainable parameters'.format(count))

# if __name__ == '__main__':
#     x = torch.randn(128, 5, 500)
#     y = base_Model()(x)
#     count_parameters(base_Model())

