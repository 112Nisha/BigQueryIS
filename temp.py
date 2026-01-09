import pandas as pd

df = pd.DataFrame({
    "age": [45, 60],
    "bmi": [28.1, 31.4],
    "text_note": [
        "Patient reports frequent thirst and fatigue",
        "No major symptoms reported"
    ]
})

df.to_parquet(
    "data/output/fused_multimodal_generic/demo.parquet",
    index=False
)

print(df)
