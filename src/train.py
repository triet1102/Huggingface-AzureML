from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import AdamW
import torch.nn as nn
from datasets import load_metric
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch
import argparse
import sys
import os
from azureml.core import Run

run = Run.get_context()


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
        '--sequence-length',
        type=int,
        default=256,
        help='Sequence length for tokenizer'
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
        '--save-model',
        type=bool,
        default=False,
        help='Whether to save the trained model'
    )

    arguments, _ = parser.parse_known_args()
    return arguments


def tokenize_function(example, tokenizer, seq_length):
    return tokenizer(example["sentence1"], example["sentence2"], max_length=seq_length, padding="max_length", truncation=True)


def main(checkpoint: str,
         learning_rate: float,
         sequence_length: int,
         batch_size: int,
         num_epoch: int,
         num_warmup: int,
         save_model: str):

    # Get PyTorch environment variables
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    distributed = world_size > 1

    print(f"DISTRIBUTED: {distributed}")
    # Set device
    if distributed:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if distributed:
        torch.distributed.init_process_group(backend="nccl")

    # Init tokenizer, model (and wrap model with DDP) and optimizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2)
    model = model.to(device)
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank
        )

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Load and preprocess dataset
    raw_datasets = load_dataset("glue", "mrpc")

    tokenized_datasets = raw_datasets.map(
        tokenize_function, fn_kwargs=dict(tokenizer=tokenizer, seq_length=sequence_length), batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(
        ["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    # Prepare train and eval dataloader
    if distributed:
        train_sampler = DistributedSampler(tokenized_datasets["train"])
    else:
        train_sampler = None

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        # shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
        sampler=train_sampler
    )

    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=batch_size,
        collate_fn=data_collator
    )

    num_training_steps = num_epoch * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=num_training_steps,
    )
    print(f"Number of training steps: {num_training_steps}")

    # # Train model and log loss
    # model.train()
    # progress_bar = tqdm(range(num_training_steps))
    # for epoch in range(num_epoch):
    #     print("Rank %d: Starting epoch %d" % (rank, epoch))
    #     if distributed:
    #         train_sampler.set_epoch(epoch)

    #     running_loss = 0.0
    #     for i, batch in enumerate(train_dataloader, 1):
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         # Forward + backward + optimize
    #         outputs = model(**batch)
    #         loss = outputs.loss
    #         loss.backward()
    #         optimizer.step()
    #         lr_scheduler.step()

    #         # Zero the parameter gradients
    #         optimizer.zero_grad()
    #         progress_bar.update(1)

    #         # Print stats
    #         running_loss += loss.item()
    #         if i % 10 == 0:
    #             log_loss = running_loss/10
    #             # Log loss metric to Azure Machine Learning
    #             # run.log("loss", log_loss)
    #             # print(f"epoch={epoch}, batch={i:5}: loss={log_loss:.2f}")
    #             print(
    #                 f"Rank:{rank}, epoch={epoch}, batch={i:5} loss={log_loss:.2f}")
    #             running_loss = 0.0

    print(f"Rank {rank}: Finished Training")

    if not distributed or rank == 0:
        if save_model:
            print("SAVING TRAINED MODEL...")
            os.makedirs("./outputs", exist_ok=True)
            torch.save(model.module.state_dict(), "outputs/model.pt")
            print(f"Rank {rank}: Finished save model")

            # Evaluate on the test dataset
            print("CALCULATING METRIC...")
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
            print(f"Rank {rank}: Finished compute metric")




if __name__ == "__main__":
    args = get_program_arguments()
    main(checkpoint=args.checkpoint,
         learning_rate=args.learning_rate,
         sequence_length=args.sequence_length,
         batch_size=args.batch_size,
         num_epoch=args.num_epoch,
         num_warmup=args.num_warmup,
         save_model=args.save_model)

    sys.exit(0)
