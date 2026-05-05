import torch
import torch.nn as nn
import torch.nn.functional as F

# Mock DWTInverse since pytorch_wavelets isn't installed in the agent's runner env
class MockDWTInverse(nn.Module):
    def __init__(self, wave='db1', mode='zero'):
        super().__init__()
    def forward(self, inputs):
        yl, yh_list = inputs
        # yl: (B, C, H, W)
        # yh: (B, C, 3, H, W)
        # Output should be (B, C, H*2, W*2)
        B, C, H, W = yl.shape
        return F.interpolate(yl, scale_factor=2, mode='nearest')

class WaveletActivation(nn.Module):
    def __init__(self, beta: float = 0.5):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(1) * beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(self.beta * x) * torch.exp(-(x ** 2) / 2)

class DirectionalSubbandAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels * 3, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, 3),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_high: torch.Tensor):
        B, C, num_subbands, H, W = x_high.shape
        pooled = x_high.mean(dim=[-2, -1])
        pooled_flat = pooled.view(B, -1)
        raw_weights = self.fc(pooled_flat)
        attn_weights = self.softmax(raw_weights)
        attn = attn_weights.view(B, 1, 3, 1, 1)
        x_high_weighted = x_high * attn
        return x_high_weighted, attn_weights

class WaveletUp(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        c_high: int = -1,
        wave: str = 'db1',
        use_dsa: bool = True,
    ):
        super().__init__()
        if c_high < 0:
            c_high = c1

        self.idwt = MockDWTInverse(wave=wave, mode='zero')

        self.align_conv = (
            nn.Sequential(
                nn.Conv2d(c1, c_high, 1, bias=False),
                nn.BatchNorm2d(c_high),
                nn.SiLU(),
            )
            if c1 != c_high else nn.Identity()
        )

        self.use_dsa = use_dsa
        if use_dsa:
            self.dsa = DirectionalSubbandAttention(channels=c_high, reduction=4)

        self.conv = nn.Conv2d(c_high, c2, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(c2)
        self.act = WaveletActivation(beta=0.5)
        self.gamma = nn.Parameter(torch.ones(1) * 0.3)
        self.last_attn = None

    def forward(
        self,
        x_low: torch.Tensor,
        x_high: torch.Tensor,
    ) -> torch.Tensor:
        x_low_aligned = self.align_conv(x_low)

        if self.use_dsa:
            x_high_weighted, attn_w = self.dsa(x_high)
            self.last_attn = attn_w.detach()
        else:
            x_high_weighted = x_high
            self.last_attn = None

        out_wavelet = self.idwt((x_low_aligned, [x_high_weighted]))

        out_standard = F.interpolate(
            x_low_aligned,
            scale_factor=2,
            mode='bilinear',
            align_corners=False,
        )

        fused = out_wavelet + (self.gamma * out_standard)
        out = self.act(self.bn(self.conv(fused)))

        return out

if __name__ == "__main__":
    torch.manual_seed(42)
    B, C, H, W = 2, 256, 40, 40

    x_low = torch.randn(B, C, H // 2, W // 2)
    x_high = torch.randn(B, C, 3, H // 2, W // 2)

    model = WaveletUp(c1=C, c2=C // 2, c_high=C, use_dsa=True)
    out = model(x_low, x_high)

    print("Success!")
    print(f"Output shape: {out.shape}")
    print(f"Attn shape: {model.last_attn.shape}")
