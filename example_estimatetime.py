import torch
import torch.nn as nn
from model_profiler import profile_flops_and_memory_layername
from model_profiler import estimate_inference_time, export_profile_to_excel_withinferencetime

# === bulid a simple CNN model ===
class CBA(nn.Module):
    '''
    conv+BN+LeakyReLU
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 k_size,
                 padding,
                 stride,
                 bias=False,
                 dilation=1):
        super(CBA, self).__init__()

        self.cba_unit = nn.Sequential(
                        nn.Conv2d(in_channels,
                                out_channels,
                                k_size,
                                padding=padding,
                                stride=stride,
                                bias=bias,
                                dilation=dilation), 
                        nn.BatchNorm2d(out_channels), 
                        nn.LeakyReLU()
                        )

    def forward(self, inputs):
        outputs = self.cba_unit(inputs)
        return outputs
    

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = CBA(in_channels=32,
                        out_channels=64,
                        k_size=3,
                        padding=1,
                        stride=2)

        self.fc = nn.Linear(64 * 8 * 8, 10)  # suppose image size is 32x32

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = SimpleCNN()

    # Profile model
    stats = profile_flops_and_memory_layername(
        model,
        input_size=(1, 3, 32, 32),
        mode="raw",   # raw / cba / block
        skip_bn=True,
        skip_act=True,
        skip_Sequential=True
    )

    # 假設一個 1 TOPS NPU + 1 GB/s DRAM + 1MB SRAM
    latency = estimate_inference_time(stats, compute_tops=1, mem_bw_gbs=1, sram_size_mb=1)
    export_profile_to_excel_withinferencetime(stats, filename="estimatetime_profile_report.xlsx",
                                             compute_tops=1, mem_bw_gbs=1, sram_size_mb=1)