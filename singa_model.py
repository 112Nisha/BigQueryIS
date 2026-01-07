import singa.tensor as tensor
import numpy as np
import pandas as pd

df = pd.read_parquet("data/output/fused_multimodal_generic")

numeric_df = df.select_dtypes(include=[np.number])
numeric_features = numeric_df.to_numpy()
text_df = df.select_dtypes(include=["object"])
text_features_raw = text_df.fillna("").astype(str).agg(" ".join, axis=1)
text_features = text_features_raw.str.len().to_numpy().reshape(-1, 1)


# Convert to SINGA tensors
num_t = tensor.Tensor(data=numeric_features)
text_t = tensor.Tensor(data=text_features)

# Simple multimodal fusion
fused = tensor.concat((num_t, text_t), axis=1)

input_dim = fused.shape[1]          # ← critical fix
weights = tensor.Tensor(
    data=np.random.rand(input_dim, 1)
)
prediction = tensor.matmul(fused, weights)

print("Multimodal predictions:")
print(prediction)
