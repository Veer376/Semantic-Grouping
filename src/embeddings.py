# from models import * 

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    print('inside the embeddings'
    )
    # inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to device
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding.cpu().numpy()

