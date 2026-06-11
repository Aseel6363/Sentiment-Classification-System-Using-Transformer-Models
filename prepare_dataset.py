import pandas as pd
from datasets import load_dataset

def main():
    """
    Downloads 100 English sentiment samples from Hugging Face IMDb dataset.
    Output file: clean_dataset.csv
    Columns: text, true_label
    """
    dataset = load_dataset("imdb", split="test[:100]")
    df = pd.DataFrame(dataset)
    df["true_label"] = df["label"].map({0: "NEGATIVE", 1: "POSITIVE"})
    df = df[["text", "true_label"]]
    df.to_csv("clean_dataset.csv", index=False, encoding="utf-8")
    print("Saved clean_dataset.csv with", len(df), "samples")

if __name__ == "__main__":
    main()
