# main.py 或你的脚本最上方
import warnings

# 忽略 pkg_resources 的弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

import torch
import torch.nn.functional as F
from umami_nnunet.network.Decoder import DynamicPSPADecoder3D, PSPADecoder3D
from umami_nnunet.network.model.VSSM import VSSM3D
import torch.nn as nn
from umami_nnunet.network.model.VSSM import VSSM3D
from umami_nnunet.network.model.LoE import ChannelAttention3D, LoE3D, LoENm3D
from torch.nn import Conv3d, InstanceNorm3d, LeakyReLU


arch_kwargs = {
    "n_stages": 4,
    "features_per_stage": [
        32,
        64,
        128,
        256,
    ],
    "conv_op": torch.nn.modules.conv.Conv3d,
    "kernel_sizes": [
        [
            3,
            3,
            3
        ],
        [
            3,
            3,
            3
        ],
        [
            3,
            3,
            3
        ],
        [
            3,
            3,
            3
        ]
    ],
    "strides": [
        [
            1,
            1,
            1
        ],
        [
            2,
            2,
            2
        ],
        [
            2,
            2,
            2
        ],
        [
            2,
            2,
            2
        ]
    ],
    "n_conv_per_stage": [
        2,
        2,
        2,
        2,
    ],
    "n_conv_per_stage_decoder": [
        2,
        2,
        2,
        2,
        2
    ],
    "conv_bias": True,
    "norm_op": torch.nn.modules.instancenorm.InstanceNorm3d,
    "norm_op_kwargs": {
        "eps": 1e-05,
        "affine": True
    },
    "dropout_op": None,
    "dropout_op_kwargs": None,
    "nonlin": torch.nn.LeakyReLU,
    "nonlin_kwargs": {
        "inplace": True
    }
}
def _pick_gn_groups(C, max_groups=8):
    for g in [max_groups, 16, 8, 4, 2, 1]:
        if g <= C and C % g == 0:
            return g
    return 1

def replace_bn3d(model: nn.Module, max_groups=8, bn1_to_identity=True, copy_affine=True):
    for name, m in model.named_children():
        if isinstance(m, nn.BatchNorm3d):
            C = m.num_features
            if C == 1 and bn1_to_identity:
                new_m = nn.Identity()
            else:
                g = _pick_gn_groups(C, max_groups=max_groups)
                new_m = nn.GroupNorm(num_groups=g, num_channels=C, eps=m.eps, affine=m.affine)

                if copy_affine and m.affine:
                    with torch.no_grad():
                        new_m.weight.copy_(m.weight)
                        new_m.bias.copy_(m.bias)

            setattr(model, name, new_m)
        else:
            replace_bn3d(m, max_groups=max_groups, bn1_to_identity=bn1_to_identity, copy_affine=copy_affine)
    return model

def replace_bn_with_in(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm3d):
            # 获取原 BN 层的通道数
            num_features = module.num_features
            # 创建新的 InstanceNorm3d 层
            # affine=True 保证保留可学习的偏移和缩放参数
            # track_running_stats=False 是 IN 的标准做法，不记录全局均值
            new_in = nn.InstanceNorm3d(num_features, affine=True, track_running_stats=False)
            
            # 替换当前层
            setattr(model, name, new_in)
        else:
            # 递归处理子模块
            replace_bn_with_in(module)

class Umami(nn.Module):
    """
    Umami 的 net parameters: 1,462,987 (1.463 M)
    """
    def __init__(self, in_c=3, num_classes=10, use_fp=False, use_ma=True, use_edl=False):
        super().__init__()

        channels=[256, 128, 64, 32]
        self.use_ma = use_ma
        self.use_edl = use_edl

        self.enc1 =  nn.Sequential(
            Conv3d(in_c, channels[3], kernel_size=3, stride=1, padding=1),
            InstanceNorm3d(channels[3], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),

            Conv3d(channels[3], channels[3], kernel_size=3, stride=1, padding=1),
            InstanceNorm3d(channels[3], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.enc2 =  nn.Sequential(
            Conv3d(channels[3], channels[2], kernel_size=3, stride=2, padding=1),
            InstanceNorm3d(channels[2], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),


            Conv3d(channels[2], channels[2], kernel_size=3, stride=1, padding=1),
            InstanceNorm3d(channels[2], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.enc3 =  nn.Sequential(
            Conv3d(channels[2], channels[1], kernel_size=3, stride=2, padding=1),
            InstanceNorm3d(channels[1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),


            Conv3d(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
            InstanceNorm3d(channels[1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.enc4 =  nn.Sequential(
            Conv3d(channels[1], channels[0], kernel_size=3, stride=2, padding=1),
            InstanceNorm3d(channels[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),

            Conv3d(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
            InstanceNorm3d(channels[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.mblk4 =  nn.Sequential(
            VSSM3D(channels[0]),
            LoENm3D(channels[0])
        )

        self.mblk3 =  nn.Sequential(
            VSSM3D(channels[1]),
            LoENm3D(channels[1])
        )

        self.mblk2 =  nn.Sequential(
            VSSM3D(channels[2]),
            LoENm3D(channels[2])
        )


        self.decoder = PSPADecoder3D()
        
        replace_bn_with_in(self.decoder)

        if use_edl:
            self.out = nn.Sequential(
                nn.Conv3d(channels[3], num_classes, kernel_size=1),
                nn.Softplus()
            )
        else:
            self.out_head4 = nn.Conv3d(channels[0], num_classes, 1) # 256 -> 10
            self.out_head3 = nn.Conv3d(channels[1], num_classes, 1)
            self.out_head2 = nn.Conv3d(channels[2], num_classes, 1)
            self.out_head1 = nn.Conv3d(channels[3], num_classes, 1) # 32 -> 10
    
    def forward(self, x, mode=None):
        if mode is None:
            mode = "train" if self.training else "eval"
        
        skip = []
        if self.use_ma:
            e1 = self.enc1(x)
            e2 = self.mblk2(self.enc2(e1))
            e3 = self.mblk3(self.enc3(e2))
            e4 = self.mblk4(self.enc4(e3))
            x = e4
            skip = [e3, e2, e1]

        else:
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)

            out = [e4, e3, e2, e1]

            x = out[0]
            skip.append(out[1])
            skip.append(out[2])
            skip.append(out[3])
        
        dec_outs = self.decoder(x, skip)
        
        if self.use_edl:
            edi = self.out(dec_outs[3])
            return edi
        else:
            if mode == "train":
                p4 = self.out_head4(dec_outs[0])
                p3 = self.out_head3(dec_outs[1])
                p2 = self.out_head2(dec_outs[2])
                p1 = self.out_head1(dec_outs[3])
            else:
                p1 = self.out_head1(dec_outs[3])

            # p4 = F.interpolate(p4, scale_factor=8, mode='trilinear')
            # p3 = F.interpolate(p3, scale_factor=4, mode='trilinear')
            # p2 = F.interpolate(p2, scale_factor=2, mode='trilinear')
            # p1 = F.interpolate(p1, scale_factor=1, mode='trilinear')

        # return [p4, p3, p2, p1]
            if mode == "train":
                return [p1, p2, p3, p4]
            else:
                return p1

class UmamiDynamic(nn.Module):
    def __init__(self, in_c=3, num_classes=10, arch_params=None):
        super().__init__()
        
        # 1. 提取 nnU-Net 的 arch_params
        features = arch_params['features_per_stage'] # [32, 64, 128, 256, 320, 320]
        strides = arch_params['strides']             # [[1,1,1], [2,2,2], ..., [1,2,2]]
        n_stages = arch_params['n_stages']           # 6

        # 提取步长并逆序，用于 Decoder 的上采样
        # 注意：Decoder 的第 i 层上采样，对应的是 Encoder 第 i+1 层的下采样步长
        decoder_strides = strides[1:][::-1] # 丢掉第一层(1,1,1)，然后逆序
        # 补一个 [1,1,1] 作为占位，确保长度匹配
        decoder_strides.append([1, 1, 1])
        
        self.encoder_stages = nn.ModuleList()
        self.mblks = nn.ModuleList()
        
        # 2. 动态构建 Encoder
        for s in range(n_stages):
            input_features = in_c if s == 0 else features[s-1]
            out_features = features[s]
            
            # 每个 stage 包含 n_conv_per_stage (2个) 卷积块
            stage = nn.Sequential(
                # 第一个卷积负责下采样 (stride)
                nn.Conv3d(input_features, out_features, kernel_size=3, 
                          stride=tuple(strides[s]), padding=1),
                nn.InstanceNorm3d(out_features, affine=True),
                nn.LeakyReLU(inplace=True),
                
                # 第二个卷积是普通的
                nn.Conv3d(out_features, out_features, kernel_size=3, 
                          stride=1, padding=1),
                nn.InstanceNorm3d(out_features, affine=True),
                nn.LeakyReLU(inplace=True)
            )
            self.encoder_stages.append(stage)
            
            # 3. 动态插入你的自定义模块 (VSSM3D / LoENm3D)
            # 根据 nnU-Net 逻辑，通常在深层 stage 插入特殊模块
            if s >= 2: # 比如在后 4 层插入
                self.mblks.append(nn.Sequential(
                    VSSM3D(out_features),
                    LoENm3D(out_features)
                ))
            else:
                self.mblks.append(nn.Identity())

        # 4. 动态构建输出头 (Deep Supervision)
        self.out_heads = nn.ModuleList([
            nn.Conv3d(f, num_classes, 1) for f in features
        ])
        
        # Plans-aware decoder for an arbitrary number of encoder stages.
        channels = features[::-1]
        self.decoder = DynamicPSPADecoder3D(channels, decoder_strides)

    def forward(self, x):
        skips = []
        feat = x
        
        # 循环执行 Encoder
        for s in range(len(self.encoder_stages)):
            feat = self.encoder_stages[s](feat)
            feat = self.mblks[s](feat)
            if s < len(self.encoder_stages) - 1:
                skips.append(feat)
        
        # Decoder 此时接收最后一个 feat 和之前的 skips
        dec_outs = self.decoder(feat, skips[::-1])[::-1] # 逆序回正
        outputs = [self.out_heads[i](dec_outs[i]) for i in range(len(dec_outs))]

        if self.training:
            return outputs
        return outputs[0]
        

class UmamiFlux(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        # 1. 初始化模型
        self.seg_model = Umami(in_c=3, num_classes=10, use_fp=True, use_ma=True)
        self.reg_head = nn.Conv3d(10, 1, kernel_size=1)

        # 2. 加载权重 (如果提供了路径)
        if checkpoint_path:
            self.load_seg_weights(checkpoint_path)

        # 3. 冻结 seg_model 的所有参数
        self.freeze_seg_model()

    def load_seg_weights(self, path):
        # map_location 确保了在不同硬件间加载的兼容性
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        
        # 如果你的权重文件里包含除了 state_dict 以外的东西（比如 epoch 等）
        # 记得根据实际情况取值，例如 state_dict['model_state_dict']
        self.seg_model.load_state_dict(state_dict['network_weights'], strict=True)
        print(f"Successfully loaded weights from {path}")

    def freeze_seg_model(self):
        for param in self.seg_model.parameters():
            param.requires_grad = False
        self.seg_model.eval() # 建议同时设为 eval 模式（影响 BatchNorm 和 Dropout）
        print("seg_model has been frozen.")

    def forward(self, x):
        # 确保在推理/前传时，seg_model 不会计算梯度（节省显存）
        seg_logits = self.seg_model(x)
        if isinstance(seg_logits, (list, tuple)):
            seg_logits = seg_logits[0]
        
        return self.reg_head(seg_logits)

class UmamiNODS(nn.Module):
    """
    Umami without deep supervision. 其 net parameters: 1,462,987 (1.463 M)，和 Umami 一样，因为只是去掉了 deep supervision 的输出头，主干完全一样。
    """
    def __init__(self, in_c=3, num_classes=10, use_fp=False, use_ma=True):
        super().__init__()

        channels=[256, 128, 64, 32]
        self.use_ma = use_ma

        self.enc1 =  nn.Sequential(
            Conv3d(in_c, channels[3], kernel_size=3, stride=1, padding=1),
            InstanceNorm3d(channels[3], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),

            Conv3d(channels[3], channels[3], kernel_size=3, stride=1, padding=1),
            InstanceNorm3d(channels[3], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.enc2 =  nn.Sequential(
            Conv3d(channels[3], channels[2], kernel_size=3, stride=2, padding=1),
            InstanceNorm3d(channels[2], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),


            Conv3d(channels[2], channels[2], kernel_size=3, stride=1, padding=1),
            InstanceNorm3d(channels[2], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.enc3 =  nn.Sequential(
            Conv3d(channels[2], channels[1], kernel_size=3, stride=2, padding=1),
            InstanceNorm3d(channels[1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),


            Conv3d(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
            InstanceNorm3d(channels[1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.enc4 =  nn.Sequential(
            Conv3d(channels[1], channels[0], kernel_size=3, stride=2, padding=1),
            InstanceNorm3d(channels[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),

            Conv3d(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
            InstanceNorm3d(channels[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.mblk4 =  nn.Sequential(
            VSSM3D(channels[0]),
            LoENm3D(channels[0])
        )

        self.mblk3 =  nn.Sequential(
            VSSM3D(channels[1]),
            LoENm3D(channels[1])
        )

        self.mblk2 =  nn.Sequential(
            VSSM3D(channels[2]),
            LoENm3D(channels[2])
        )


        self.decoder = PSPADecoder3D()
        
        replace_bn_with_in(self.decoder)

        self.out_head1 = nn.Conv3d(channels[3], num_classes, 1) # 32 -> 10
    
    def forward(self, x, mode="train"):
        
        skip = []
        if self.use_ma:
            e1 = self.enc1(x)
            e2 = self.mblk2(self.enc2(e1))
            e3 = self.mblk3(self.enc3(e2))
            e4 = self.mblk4(self.enc4(e3))
            x = e4
            skip = [e3, e2, e1]

        else:
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)

            out = [e4, e3, e2, e1]

            x = out[0]
            skip.append(out[1])
            skip.append(out[2])
            skip.append(out[3])
        
        dec_outs = self.decoder(x, skip)

        p1 = self.out_head1(dec_outs[3])

        return p1

class UmamiRefine(nn.Module):
    """
    Umami 的 net parameters: 1,462,987 (1.463 M)
    将最终的输出前多加一个 refine 模块，进行最后的特征融合和预测。
    """
    def __init__(self, in_c=3, num_classes=10):
        super().__init__()

        channels=[256, 128, 64, 32]

        self.enc1 =  nn.Sequential(
            Conv3d(in_c, channels[3], kernel_size=3, stride=1, padding=1),
            InstanceNorm3d(channels[3], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),

            Conv3d(channels[3], channels[3], kernel_size=3, stride=1, padding=1),
            InstanceNorm3d(channels[3], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.enc2 =  nn.Sequential(
            Conv3d(channels[3], channels[2], kernel_size=3, stride=2, padding=1),
            InstanceNorm3d(channels[2], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),


            Conv3d(channels[2], channels[2], kernel_size=3, stride=1, padding=1),
            InstanceNorm3d(channels[2], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.enc3 =  nn.Sequential(
            Conv3d(channels[2], channels[1], kernel_size=3, stride=2, padding=1),
            InstanceNorm3d(channels[1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),


            Conv3d(channels[1], channels[1], kernel_size=3, stride=1, padding=1),
            InstanceNorm3d(channels[1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.enc4 =  nn.Sequential(
            Conv3d(channels[1], channels[0], kernel_size=3, stride=2, padding=1),
            InstanceNorm3d(channels[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),

            Conv3d(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
            InstanceNorm3d(channels[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.mblk4 =  nn.Sequential(
            VSSM3D(channels[0]),
            LoENm3D(channels[0])
        )

        self.mblk3 =  nn.Sequential(
            VSSM3D(channels[1]),
            LoENm3D(channels[1])
        )

        self.mblk2 =  nn.Sequential(
            VSSM3D(channels[2]),
            LoENm3D(channels[2])
        )


        self.decoder = PSPADecoder3D()
        
        replace_bn_with_in(self.decoder)

        self.out_head4 = nn.Conv3d(channels[0], num_classes, 1) # 256 -> 10
        self.out_head3 = nn.Conv3d(channels[1], num_classes, 1)
        self.out_head2 = nn.Conv3d(channels[2], num_classes, 1)
        # self.out_head1 = nn.Conv3d(channels[3], num_classes, 1) # 32 -> 10
        self.out_head1 = nn.Sequential(
            nn.Conv3d(channels[3], channels[3], kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(channels[3], affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(channels[3], num_classes, kernel_size=1)
        )
    
    def forward(self, x, mode=None):
        if mode is None:
            mode = "train" if self.training else "eval"
        
        skip = []
        e1 = self.enc1(x)
        e2 = self.mblk2(self.enc2(e1))
        e3 = self.mblk3(self.enc3(e2))
        e4 = self.mblk4(self.enc4(e3))
        x = e4
        skip = [e3, e2, e1]
        
        dec_outs = self.decoder(x, skip)
        
        p4 = self.out_head4(dec_outs[0])
        p3 = self.out_head3(dec_outs[1])
        p2 = self.out_head2(dec_outs[2])
        p1 = self.out_head1(dec_outs[3])


        # return [p4, p3, p2, p1]
        if mode == "train":
            return [p1, p2, p3, p4]
        else:
            return p1

if __name__ == '__main__':
    from torchinfo import summary
    import torch

    # model = UnetCADFPSub(in_c=3, num_classes=10)  # 你的模型
    x = torch.randn(1, 3, 128, 128, 128).cuda()
    model = model.cuda()

    s = summary(model, input_data=x, depth=6, verbose=1)
    print(s)

    # 保存到 txt
    # with open("model_summary.txt", "w", encoding="utf-8") as f:
    #     f.write(str(s))


    # print(net)

    # total_params = sum(p.numel() for p in net.parameters())
    # print(f"net parameters: {total_params:,} ({total_params/1e6:.3f} M)")

    # # # 输入
    # img = torch.randn((1,3,128,128,128)).to(device)
    
    # # 输出 torch.Size([4, 10, 128, 128, 128]), logit
    # out = net.forward(img)
    # for feat in out:
    #     print("feat", feat.shape)
