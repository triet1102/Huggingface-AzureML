from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from datasets import load_dataset
import os
from transformers import AutoConfig
from onnxruntime import InferenceSession

checkpoint = r"bert-base-uncased"

# # Save model in pt

# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForSequenceClassification.from_pretrained(
#     checkpoint, num_labels=2)

# print(os.getcwd())
# os.makedirs("./trash_dir", exist_ok=True)
# torch.save(model.state_dict(), "./trash_dir/model.pt")


# # Load pt model
# config = AutoConfig.from_pretrained('bert-base-uncased')

# model =  AutoModelForSequenceClassification.from_config(config)
# model.load_state_dict(torch.load("./trash_dir/model.pt"))
# model.save_pretrained("trash_bert_base_classification")

# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# tokenizer.save_pretrained("trash_bert_base_classification")

# print("OK")

# Command to save onnx model
# python -m transformers.onnx --model=trash_bert_base_classification onnx/


# # Test onnx model
# session = InferenceSession("onnx/model.onnx")
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# inputs = tokenizer("Using DistilBERT with ONNX Runtime!", return_tensors="pt")
# outputs = session.run(output_names=["last_hidden_state"], input_feed=inputs)
# print('ok')

raw_datasets = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=2)

s1_1 = "The identical rovers will act as robotic geologists , searching for evidence of past water ."
s2_1 = "The rovers act as robotic geologists , moving on six wheels ."
txt_tokenized = tokenizer(s1_1, s2_1, return_tensors="pt")
outputs = model(**txt_tokenized)
print(f"output_1: {outputs}")


s1_2 = "Spider-Man snatched $ 114.7 million in its debut last year and went on to capture $ 403.7 million ."
s2_2 = "Spider-Man , rated PG-13 , snatched $ 114.7 million in its first weekend and went on to take in $ 403.7 million ."
txt_tokenized = tokenizer(s1_2, s2_2, return_tensors="pt")
outputs = model(**txt_tokenized)
print(f"output_2: {outputs}")
