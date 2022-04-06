from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
from collections import OrderedDict


def main(model_path,
         save_pretrained_dir):
    # Init SequenceClassification model with random weights
    checkpoint = "bert-base-uncased"
    config = AutoConfig.from_pretrained(checkpoint)
    config.num_labels = 2
    model = AutoModelForSequenceClassification.from_config(config)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    model.save_pretrained(save_pretrained_dir)
    tokenizer.save_pretrained(save_pretrained_dir)
    print("Completed save pretrained model")


if __name__ == "__main__":
    model_path = "model/model.pt"
    save_pretrained_dir = "model/BertSentenceClassification"

    main(model_path=model_path,
         save_pretrained_dir=save_pretrained_dir)
