from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from datasets import load_metric
from transformers import AutoConfig
import torch
from collections import OrderedDict


def tokenize_function(example, tokenizer, seq_length):
    return tokenizer(example["sentence1"], example["sentence2"], max_length=seq_length, padding="max_length", truncation=True)


def main(checkpoint,
         sequence_length):
    raw_datasets = load_dataset("glue", "mrpc")

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    tokenized_datasets = raw_datasets.map(
        tokenize_function, fn_kwargs=dict(tokenizer=tokenizer, seq_length=sequence_length), batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(
        ["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )

    metric = load_metric("glue", "mrpc")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    print(f"Metric computed: \n{metric.compute()}")


if __name__ == "__main__":
    checkpoint = "./model/BertSentenceClassification"
    sequence_length = 256
    main(checkpoint,
         sequence_length)
