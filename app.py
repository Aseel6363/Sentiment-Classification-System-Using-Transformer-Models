import streamlit as st
from transformers import pipeline
from label_utils import unify_label

MODELS = {
    "DistilBERT": "distilbert-base-uncased-finetuned-sst-2-english",
    "RoBERTa": "cardiffnlp/twitter-roberta-base-sentiment",
    "BERT": "nlptown/bert-base-multilingual-uncased-sentiment",
}

st.set_page_config(page_title="Sentiment Classification System")

st.title("Sentiment Classification System")
st.write("Using pretrained Hugging Face models")

@st.cache_resource
def load_model(model_name):
    model_id = MODELS[model_name]
    return pipeline("sentiment-analysis", model=model_id, tokenizer=model_id)

sentence = st.text_area("Enter a sentence:", height=120)
model_name = st.selectbox("Select Model:", list(MODELS.keys()))

if st.button("Predict"):
    if not sentence.strip():
        st.warning("Please enter a sentence first.")
    else:
        classifier = load_model(model_name)
        result = classifier(sentence[:512])[0]
        label = unify_label(model_name, result["label"])
        confidence = result["score"] * 100

        st.success(f"Predicted Label: {label}")
        st.metric("Confidence Score", f"{confidence:.2f}%")

st.divider()

if st.checkbox("Comparison View: Run all three models"):
    if not sentence.strip():
        st.info("Enter a sentence above to compare all models.")
    else:
        rows = []
        for name in MODELS:
            classifier = load_model(name)
            result = classifier(sentence[:512])[0]
            rows.append({
                "Model": name,
                "Predicted Label": unify_label(name, result["label"]),
                "Confidence": f"{result['score'] * 100:.2f}%"
            })
        st.table(rows)
