# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
from config import CONF

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class AlphaGomokuNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.start_block = nn.Sequential(
            nn.Conv2d(CONF.INPUT_CHANNELS, CONF.NUM_FILTERS, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(CONF.NUM_FILTERS),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.ModuleList([ResBlock(CONF.NUM_FILTERS) for _ in range(CONF.NUM_RES_BLOCKS)])
        self.policy_conv = nn.Conv2d(CONF.NUM_FILTERS, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * CONF.BOARD_SIZE**2, CONF.ACTION_SIZE)
        self.value_conv = nn.Conv2d(CONF.NUM_FILTERS, 16, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * CONF.BOARD_SIZE**2, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self._initialize_weights()

    def forward(self, x):
        if x.dtype != torch.float32: x = x.to(torch.float32)
        x = self.start_block(x)
        for block in self.res_blocks: x = block(x)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return p, v

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

def save_model(model, path):
    try: torch.save(model.state_dict(), path)
    except Exception as e: logging.error(f"Save failed: {e}")

def load_model(model, path, device):
    try:
        state = torch.load(path, map_location=device)
        if 'model_state_dict' in state: state = state['model_state_dict']
        model.load_state_dict(state)
        return True
    except Exception as e:
        logging.warning(f"Load failed: {e}")
        return False