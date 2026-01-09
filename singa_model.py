import numpy as np
import pandas as pd
from singa import tensor, device, autograd

dev = device.create_cpu_device()

df = pd.read_parquet("data/output/fused_multimodal_generic/part-00000-1967448f-53ce-4fb3-9646-3b380f10ef84-c000.snappy.parquet")

numeric = df[["age", "bmi"]].to_numpy().astype(np.float32)
text_len = (
    df["text_note"].astype(str)
    .str.len()
    .to_numpy()
    .reshape(-1, 1)
    .astype(np.float32)
)

num_t = tensor.Tensor(device=dev, data=numeric)
text_t = tensor.Tensor(device=dev, data=text_len)

fused = autograd.Concat(axis=1)(num_t, text_t)
fused = fused[0]  # unwrap

fused = tensor.Tensor(
    device=dev,
    data=tensor.to_numpy(fused).astype(np.float32)
)

weights = tensor.Tensor(
    device=dev,
    data=np.array([[0.02], [0.5], [0.01]], dtype=np.float32)
)

bias = tensor.Tensor(
    device=dev,
    data=np.array([[0.1]], dtype=np.float32)  # make bias 2D
)

linear = autograd.Matmul()(fused, weights)
linear = linear[0]

score = autograd.Add()(linear, bias)
score = score[0]

print("\n=== Multimodal Singa Output ===")
print(tensor.to_numpy(score))
