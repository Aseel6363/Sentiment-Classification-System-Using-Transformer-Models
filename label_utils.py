def unify_label(model_name: str, raw_label: str) -> str:
    """
    Convert different Hugging Face model outputs into POSITIVE / NEGATIVE.
    """
    label = str(raw_label).upper().strip()

    # DistilBERT SST-2 returns POSITIVE / NEGATIVE
    if label in ["POSITIVE", "NEGATIVE"]:
        return label

    # Cardiff RoBERTa returns LABEL_0 negative, LABEL_1 neutral, LABEL_2 positive
    if "ROBERTA" in model_name.upper():
        if label == "LABEL_0":
            return "NEGATIVE"
        if label == "LABEL_2":
            return "POSITIVE"
        if label == "LABEL_1":
            return "NEGATIVE"  # neutral is mapped to negative for binary comparison

    # NLP Town BERT returns 1 star ... 5 stars
    if "STAR" in label:
        number = int(label.split()[0])
        return "POSITIVE" if number >= 4 else "NEGATIVE"

    # fallback
    if label in ["LABEL_1", "1"]:
        return "POSITIVE"
    return "NEGATIVE"
