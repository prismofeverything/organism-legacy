import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F

class OrganismNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.args = args

        super(OrganismNNet, self).__init__()
        self.conv1 = HexaConv2d(11, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = HexaConv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = HexaConv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv4 = HexaConv2d(args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 11, self.board_x, self.board_y)               # batch_size x 11 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        v = self.fc3(s)                                                                          # batch_size x 1

        return torch.tanh(v)


class HexaConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):

        assert kernel_size % 2 == 1

        self.mask = torch.zeros((kernel_size, kernel_size))

        for i in range(kernel_size):
            for j in range(kernel_size):
                if abs(i + j - (kernel_size - 1)) <= (kernel_size // 2):
                    self.mask[i, j] = 1

        super(HexaConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias, padding_mode)

    def forward(self, x):
        mask = self.mask # a matrix with the masking pattern
        mask = mask[None, None] # mask is 2d, unsqueeze for broadcast

        return F.conv2d(x, self.weight * mask, self.bias, self.stride,
            self.padding, self.dilation, self.groups)