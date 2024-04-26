from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import os

app = FastAPI()

# Fetch the model directory from the environment variable
model_directory = os.getenv('MODEL_PATH', '/app/models')
model_filename = 'qa_model.pth'  # Specify your model filename here

# Full path to the model file
model_path = os.path.join(model_directory, model_filename)

# Load the model and tokenizer
model = BertForQuestionAnswering.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class QAInput(BaseModel):
    question: str
    context: str

@app.post("/predict")
async def predict(data: QAInput):
    # Encode the inputs
    inputs = tokenizer.encode_plus(data.question, data.context, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits

    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    # Convert tokens to answer string
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))

    return {"question": data.question, "context": data.context, "answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
