# 📊 model_profiler

A lightweight PyTorch model profiler that reports **FLOPs**, **memory usage**, **parameters**, **input/output shapes**, and automatically exports results to **Excel** with colored tags.

It also integrates with **Torchview** to generate computation graph diagrams.

---

## 🚀 Features
- Profile per-layer **FLOPs, memory (bytes), parameters, input/output shapes**.
- Support three modes:
  - **raw**: list all layers (Conv, BN, ReLU, …).
  - **cba**: merge `Conv+BN+Activation` into **CBA** blocks.
  - **block**: merge into higher-level blocks (e.g., backbone / neck / head).
- Export results to **Excel** with:
  - Full profiling table.
  - Color-coded rows:
    - 🔴 Memory-bound
    - 🟢 Compute-bound
    - ⚪ Balanced
  - Automatic statistics sheet (summary counts & totals).
- Generate **graph PNG** with Torchview.

---

## 📦 Installation

Clone the repository and install:
```bash
git clone https://github.com/yourname/model_profiler.git
cd model_profiler
pip install -e .
```


## 🛠 Usage
Example Code:
```python
import torch
import torch.nn as nn
from model_profiler import (
    profile_flops_and_memory_layername,
    export_profile_to_excel,
    draw_model_with_tags
)

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool  = nn.MaxPool2d(2)
        self.fc    = nn.Linear(16*16*16, 10)

    def forward(self, x):
        x = self.pool(self.relu1(self.bn1(self.conv1(x))))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Create model
model = SimpleCNN()

# Run profiler
stats = profile_flops_and_memory_layername(model, input_size=(1, 3, 32, 32), mode="cba")

# Export to Excel
export_profile_to_excel(stats, "cnn_profile.xlsx")

# Draw graph (requires Graphviz installed)
draw_model_with_tags(model, (1, 3, 32, 32), stats, filename="cnn_graph")
```

📊 Excel Output Example

| Layer(Name)       | Input Shape     | Output Shape    | FLOPs (M) | Memory (KB) | FLOP/Byte | Params (K) | Tag            |
| ----------------- | --------------- | --------------- | --------- | ----------- | --------- | ---------- | -------------- |
| conv1 (Conv2d)    | (1, 3, 32, 32)  | (1, 16, 32, 32) | 0.29      | 12.3        | 23.5      | 448        | Balanced       |
| bn1 (BatchNorm2d) | (1, 16, 32, 32) | (1, 16, 32, 32) | 0.03      | 8.0         | 0.5       | 32         | Memory-bound ❗ |
| relu1 (ReLU)      | (1, 16, 32, 32) | (1, 16, 32, 32) | 0.02      | 8.0         | 0.2       | 0          | Memory-bound ❗ |
| pool (MaxPool2d)  | (1, 16, 32, 32) | (1, 16, 16, 16) | 0.01      | 4.0         | 0.1       | 0          | Memory-bound ❗ |
| fc (Linear)       | (1, 4096)       | (1, 10)         | 0.04      | 16.0        | 2.5       | 40         | Balanced       |


📌 Roadmap

 Add CUDA memory usage profiling

 Add latency measurement

 Add visualization in Jupyter Notebook



## Dependencies:

```bash
pip install torch torchvision prettytable openpyxl torchview graphviz
```
⚠️ You must also install the Graphviz system package (see Graphviz Setup)

IF there are any problems in ```draw_model_with_tags```, please check Graphviz Executable.

⚠️ Graphviz Executable Not Found

If you see an error like this:

```
graphviz.backend.execute.ExecutableNotFound: failed to execute WindowsPath('dot'),
make sure the Graphviz executables are on your systems' PATH
```

✅ Cause

The Python ```graphviz``` package is only a binding. It requires the actual Graphviz executables (especially ```dot```) to be installed on your system and available in the PATH.

🔧 Solution
1. Download Graphviz
Get the Windows installer from the official site:
👉 https://graphviz.gitlab.io/_pages/Download/Download_windows.html
Use the stable release installer (EXE).

2. Install Graphviz
The default installation path is usually:
```
C:\Program Files\Graphviz\bin
```
3. Add Graphviz to PATH
Open Windows search → type Environment Variables.
Edit the System Environment Variables → Path.
Add:
```
C:\Program Files\Graphviz\bin
```
Click OK to save and close.
4. Restart your terminal
(Anaconda Prompt / CMD / PowerShell) so the new PATH takes effect.


📄 License

MIT License © 2025 Tommy Huang