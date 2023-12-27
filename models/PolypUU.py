import torch
import torch.nn as nn
from torch.nn.functional import feature_alpha_dropout
import torchvision.models as models


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super(DecoderBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x


class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, kernel_size=3, stride=1, padding=1):
        super(SideoutBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.dropout = nn.Dropout2d(dropout)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x


class encoder(nn.Module):
    def __init__(self, num_classes):
        super(encoder, self).__init__()

        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Encoder
        self.encoder1_conv = resnet.conv1
        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

    def forward(self, x):
        # x 224
        e1 = self.encoder1_conv(x)  # 128
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_pool = self.maxpool(e1)  # 56
        e2 = self.encoder2(e1_pool)
        e3 = self.encoder3(e2)  # 28
        e4 = self.encoder4(e3)  # 14
        e5 = self.encoder5(e4)  # 7

        return e1, e2, e3, e4, e5


class PolypUU(nn.Module):
    def __init__(self, num_classes, feat_level=4, dropout=0.1):
        super(PolypUU, self).__init__()

        self.encoder1 = encoder(num_classes)

        # Decoder
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=512 + 256, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256 + 128, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128 + 64, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64 + 64, out_channels=64)

        self.outconv = nn.Sequential(
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, num_classes, 1),
        )

        # Sideout
        self.feat_level = feat_level
        if feat_level == 0:
            pass
        else:
            self.sideout = SideoutBlock(64 * (2 * feat_level), 1, dropout=dropout)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder1(x)
        out1, feat = None, None

        if self.feat_level == 0:
            d5 = self.decoder5(e5)  # 14
            d4 = self.decoder4(torch.cat((d5, e4), dim=1))  # 28
            d3 = self.decoder3(torch.cat((d4, e3), dim=1))  # 56
            d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # 128
            d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64
            out1 = self.outconv(d1)  # 224
            feat = out1
        elif self.feat_level == 1:
            d5 = self.decoder5(e5)  # 14
            d4 = self.decoder4(torch.cat((d5, e4), dim=1))  # 28
            d3 = self.decoder3(torch.cat((d4, e3), dim=1))  # 56
            d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # 128
            # d2_ = torch.nn.functional.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
            d2_ = d2
            feat = self.sideout(d2_)
            d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64
            out1 = self.outconv(d1)  # 224
        elif self.feat_level == 2:
            d5 = self.decoder5(e5)  # 14
            d4 = self.decoder4(torch.cat((d5, e4), dim=1))  # 28
            d3 = self.decoder3(torch.cat((d4, e3), dim=1))  # 56
            # d3_ = torch.nn.functional.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=True)
            d3_ = d3
            feat = self.sideout(d3_)
            d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # 128
            d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64
            out1 = self.outconv(d1)  # 224
        elif self.feat_level == 3:
            d5 = self.decoder5(e5)  # 14
            d4 = self.decoder4(torch.cat((d5, e4), dim=1))  # 28
            # d4_ = torch.nn.functional.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=True)
            d4_ = d4
            feat = self.sideout(d4_)
            d3 = self.decoder3(torch.cat((d4, e3), dim=1))  # 56
            d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # 128
            d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64
            out1 = self.outconv(d1)  # 224
        elif self.feat_level == 4:
            d5 = self.decoder5(e5)  # 14
            # d5_ = torch.nn.functional.interpolate(d5, scale_factor=16, mode='bilinear', align_corners=True)
            d5_ = d5
            feat = self.sideout(d5_)
            d4 = self.decoder4(torch.cat((d5, e4), dim=1))  # 28
            d3 = self.decoder3(torch.cat((d4, e3), dim=1))  # 56
            d2 = self.decoder2(torch.cat((d3, e2), dim=1))  # 128
            d1 = self.decoder1(torch.cat((d2, e1), dim=1))  # 224*224*64
            out1 = self.outconv(d1)  # 224

        return torch.sigmoid(out1), torch.sigmoid(feat)
