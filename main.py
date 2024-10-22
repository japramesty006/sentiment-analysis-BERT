from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import uvicorn

app = FastAPI()

model_path = "finetuned_BERT.pth"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

label_dict_inverse = {
    0: "neutral",
    1: "negative",
    2: "positive"
}

class SentimentRequest(BaseModel):
    text: str

# Prediction function
def predict_sentiment(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Predict without calculating gradients
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()

    predicted_category = label_dict_inverse[predicted_label]

    return predicted_category
    
# API route
@app.post("/sentiment")
async def analyze_sentiment(request: SentimentRequest):
    sentiment_category = predict_sentiment(request.text)
    return {"sentiment": sentiment_category}

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
