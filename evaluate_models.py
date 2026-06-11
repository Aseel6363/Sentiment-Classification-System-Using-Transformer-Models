import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from label_utils import unify_label

MODELS = {
    "DistilBERT": "distilbert-base-uncased-finetuned-sst-2-english",
    "RoBERTa": "cardiffnlp/twitter-roberta-base-sentiment",
    "BERT": "nlptown/bert-base-multilingual-uncased-sentiment",
}

def main():
    df = pd.read_csv("clean_dataset.csv")
    metrics_rows = []

    for model_name, model_id in MODELS.items():

        print(f"\nLoading {model_name}: {model_id}")

        clf = pipeline(
            "sentiment-analysis",
            model=model_id,
            tokenizer=model_id
        )

        print("Current model =", model_id)

        preds = []
        scores = []

        for text in tqdm(
            df["text"].astype(str),
            desc=f"Predicting with {model_name}"
        ):
            result = clf(text[:512])[0]

            pred = unify_label(model_name, result["label"])

            preds.append(pred)
            scores.append(round(float(result["score"]), 4))

        df[f"{model_name}_pred"] = preds
        df[f"{model_name}_score"] = scores

        accuracy = accuracy_score(df["true_label"], preds)

        precision, recall, f1, _ = precision_recall_fscore_support(
            df["true_label"],
            preds,
            average="binary",
            pos_label="POSITIVE",
            zero_division=0
        )

        metrics_rows.append({
            "Model": model_name,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-score": round(f1, 4),
        })

    df.to_csv(
        "predictions.csv",
        index=False,
        encoding="utf-8"
    )

    metrics_df = pd.DataFrame(metrics_rows)

    metrics_df.to_csv(
        "metrics_results.csv",
        index=False,
        encoding="utf-8"
    )

    print("\nMetrics Results:")
    print(metrics_df)

if __name__ == "__main__":
    main()