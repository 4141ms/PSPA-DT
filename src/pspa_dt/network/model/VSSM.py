""" CVPR 2025 SaMam"""
import torch.nn as nn
from pspa_dt.network.model.archi_utils import PatchEmbed,PatchUnEmbed, PatchEmbed3D, PatchUnEmbed3D


from pspa_dt.network.model.SS2D_Encoder import SS3D_encoder

class VSSM3D(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 64,
            d_state: int = 16,
            expand: float = 2.,
            mamba_from_trion=1,
            **kwargs,
    ):
        super().__init__()
        self.SSM = SS3D_encoder(d_model=hidden_dim, d_state=d_state,expand=expand,mamba_from_trion=mamba_from_trion, **kwargs)

        self.patch_embed = PatchEmbed3D()
        self.patch_unembed = PatchUnEmbed3D()

    def forward(self, input):
        B, C, D, H, W = input.size()
        x = self.patch_embed(input)  # B,L(DHW),C
        # B, L, C = x.shape
        x = x.view(B, D, H, W, C).contiguous()  # x -> [B,D,H,W,C]
        x = self.SSM(x)
        x = x.view(B, -1, C).contiguous()
        x = self.patch_unembed(x, C, D, H, W)

        return x + input



if __name__ == '__main__':
    import torch

    # img = torch.randn((1, 3, 128, 128, 128)).cuda()
    img = torch.randn((1, 64, 32, 32, 32)).cuda()
    net = VSSM3D(hidden_dim=64,expand=2).cuda()

    total_params = sum(p.numel() for p in net.parameters())

    print(f"net parameters: {total_params:,} ({total_params/1e6:.3f} M)")

    out = net.forward(img)
    print(out.shape)






