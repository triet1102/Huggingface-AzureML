from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import argparse
import sys


def get_program_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        type=str,
        default="bert-base-uncased",
        help='Path to a local model or name of Huggingface model'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-5,
        help='Learning rate for training Bert sentence classification'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for training and inference'
    )
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=3,
        help='Number of epoch for training'
    )
    parser.add_argument(
        '--num-warmup',
        type=int,
        default=0,
        help='Number warmup step in training'
    )
    parser.add_argument(
        '--save-model-dir',
        type=str,
        help='Where to save the trained model'
    )

    arguments, _ = parser.parse_known_args()
    return arguments 


def tokenize_function(example, tokenizer):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


def main(checkpoint: str,
         learning_rate: float,
         batch_size: int,
         num_epoch: int,
         num_warmup: int,
         save_model_dir: str):

    checkpoint = checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    raw_datasets = load_dataset("glue", "mrpc")
    tokenized_datasets = raw_datasets.map(
        tokenize_function, fn_kwargs=dict(tokenizer=tokenizer), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(
        ["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=batch_size, collate_fn=data_collator
    )

    num_training_steps = num_epoch * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=num_training_steps,
    )
    print(f"Number of training steps: {num_training_steps}")

    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(num_epoch):
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    print("Finished Training")

    if save_model_dir:
        model.save_pretrained(save_model_dir)
        tokenizer.save_pretrained(save_model_dir)
        print("Finished save model and tokenizer")


if __name__ == "__main__":
    args = get_program_arguments()
    main(checkpoint=args.checkpoint,
         learning_rate=args.learning_rate,
         batch_size=args.batch_size,
         num_epoch=args.num_epoch,
         num_warmup=args.num_warmup,
         save_model_dir=args.save_model_dir)

    sys.exit(0)
