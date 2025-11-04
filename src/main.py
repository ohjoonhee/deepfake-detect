from dataclasses import dataclass, field
from typing import Optional, List

from datasets import load_dataset
from transformers import AutoProcessor, AutoConfig
from trl import SFTTrainer, SFTConfig, TrlParser, ModelConfig

from transformers.integrations.integration_utils import is_wandb_available

import numpy as np

import evaluate


@dataclass
class ScriptArguments:
    """
    Arguments common to all scripts.

    Args:
        dataset_name (`str`,, *optional*):
            Path or name of the dataset to load. If `datasets` is provided, this will be ignored.
        dataset_config (`str`, *optional*):
            Dataset configuration name. Corresponds to the `name` argument of the [`~datasets.load_dataset`] function.
            If `datasets` is provided, this will be ignored.
        dataset_train_split (`str`, *optional*, defaults to `"train"`):
            Dataset split to use for training. If `datasets` is provided, this will be ignored.
        dataset_test_split (`str`, *optional*, defaults to `"test"`):
            Dataset split to use for evaluation. If `datasets` is provided, this will be ignored.
        dataset_streaming (`bool`, *optional*, defaults to `False`):
            Whether to stream the dataset. If True, the dataset will be loaded in streaming mode. If `datasets` is
            provided, this will be ignored.
        gradient_checkpointing_use_reentrant (`bool`, *optional*, defaults to `False`):
            Whether to apply `use_reentrant` for gradient checkpointing.
        ignore_bias_buffers (`bool`, *optional*, defaults to `False`):
            Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid scalar
            type, inplace operation. See
            https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path or name of the dataset to load. If `datasets` is provided, this will be ignored."},
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset configuration name. Corresponds to the `name` argument of the `datasets.load_dataset` " "function. If `datasets` is provided, this will be ignored."},
    )
    train_split: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset split to use for training. If `datasets` is provided, this will be ignored."},
    )
    eval_split: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset split to use for evaluation. If `datasets` is provided, this will be ignored."},
    )
    train_eval_split: Optional[float] = field(
        default=None,
        metadata={"help": "If specified, a portion of the training set will be set aside for evaluation."},
    )
    max_pixels: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of pixels for image resizing."},
    )
    min_pixels: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of pixels for image resizing."},
    )
    dataset_streaming: bool = field(
        default=False,
        metadata={"help": "Whether to stream the dataset. If True, the dataset will be loaded in streaming mode. If " "`datasets` is provided, this will be ignored."},
    )
    gradient_checkpointing_use_reentrant: bool = field(
        default=False,
        metadata={"help": "Whether to apply `use_reentrant` for gradient checkpointing."},
    )
    ignore_bias_buffers: bool = field(
        default=False,
        metadata={
            "help": "Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid "
            "scalar type, inplace operation. See "
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992."
        },
    )


def preprocess(example):
    label_map = {0: "Fake", 1: "Real"}
    label = label_map[example["label"]]

    prompt = [
        {
            "role": "user",
            # "content": [
            #     {"type": "image", "image": example["image"]},
            #     {"type": "text", "text": "Is this image real or fake? Answer with one word: Real or Fake."},
            # ],
            "content": "Is this image real or fake? Answer with one word: Real or Fake.",
        },
    ]
    completion = [
        {"role": "assistant", "content": f"{label}."},
    ]

    # messages = [
    #     {
    #         "role": "user",
    #         # "content": [
    #         #     {"type": "image", "image": example["image"]},
    #         #     {"type": "text", "text": "Is this image real or fake? Answer with one word: Real or Fake."},
    #         # ],
    #         "content": "Is this image real or fake? Answer with one word: Real or Fake.",
    #     },
    #     {"role": "assistant", "content": f"{label}."},
    # ]

    return {"prompt": prompt, "completion": completion, "images": [example["image"]]}
    # return {"messages": messages}


def load_model_from_config(model_args: ModelConfig):
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    print("Model config:", config)
    arch = config.architectures[0] if config.architectures else "Unknown"

    # import arch and instantiate model accordingly
    import importlib

    module = importlib.import_module("transformers")
    model_class = getattr(module, arch, None)
    if model_class is None:
        raise ValueError(f"Model architecture {arch} not found in transformers library.")
    print(f"Loading model class: {model_class}")
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        dtype=model_args.dtype,
    )

    # Peft config
    if model_args.use_peft:
        from peft import get_peft_model, LoraConfig

        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)
        print("Loaded PEFT model with config:", peft_config)
        model.print_trainable_parameters()

    return model


def parse_args():
    parser = TrlParser([ScriptArguments, SFTConfig, ModelConfig])
    script_args, training_args, model_args = parser.parse_args_and_config(fail_with_unknown_args=False)  # TODO: change to True
    return script_args, training_args, model_args


def train():
    """
    Train a model using SFTTrainer and a config file.
    """
    script_args, training_args, model_args = parse_args()
    print("Script arguments:", script_args)

    # 1. Load processor
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    processor.image_processor.size = {"longest_edge": script_args.max_pixels, "shortest_edge": script_args.min_pixels}

    model = load_model_from_config(model_args)

    # 3. Load dataset
    train_dataset = load_dataset(script_args.dataset_name, split=script_args.train_split)
    train_dataset = train_dataset.map(preprocess, batched=True, batch_size=8, num_proc=8, desc="Preprocessing dataset")
    if script_args.train_eval_split is not None:
        train_dataset, eval_dataset = train_dataset.train_test_split(test_size=script_args.train_eval_split, stratify_by_column="label").values()
    else:
        eval_dataset = None

    print("Train dataset size:", train_dataset)
    print("Eval dataset size:", eval_dataset)

    # 4. Load evaluation metric
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    if eval_dataset is not None:

        class MetricAggregator:
            def __init__(self):
                self.correct = 0
                self.total = 0

            def add_batch(self, preds, labels):
                self.correct += sum(p == l for p, l in zip(preds, labels))
                self.total += len(labels)

            def compute(self):
                return {"accuracy": self.correct / self.total if self.total > 0 else 0.0}

        aggregator = MetricAggregator()

        def compute_metrics(eval_preds, compute_result: bool):
            # preds, labels = eval_preds
            preds = eval_preds.predictions
            labels = eval_preds.label_ids
            inputs = eval_preds.inputs.input_ids

            if isinstance(preds, tuple):
                preds = preds[0]

            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            inputs = inputs.cpu().numpy()

            # Replace -100 in the preds as we can't decode them
            preds = np.where(preds != -100, preds, processor.tokenizer.pad_token_id)
            # preds = [p[len(i) :] for (p, i) in zip(preds, inputs)]
            print(preds)

            # Decode generated summaries into text
            decoded_preds = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)

            # Replace -100 in the labels as we can't decode them
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            # Decode reference summaries into text
            decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
            # ROUGE expects a newline after each sentence
            print("===== After Decoding =====")
            print([len(i) for i in inputs])
            print(decoded_preds)
            # decoded_preds = [pred.strip()[len(i) :] for (pred, i) in zip(decoded_preds, inputs)]
            decoded_preds = [pred.strip() for (pred) in decoded_preds]

            decoded_labels = [label.strip() for label in decoded_labels]

            aggregator.add_batch(decoded_preds, decoded_labels)

            # Debug prints
            print("Decoded preds:", decoded_preds)
            print("Decoded labels:", decoded_labels)

            if compute_result:
                return aggregator.compute()

    else:
        compute_metrics = None

    # 4. Create Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # 5. Train
    trainer.train()

    # 6. Save
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    import sys

    sys.argv += ["--config", "configs/dev.yaml"]

    if is_wandb_available():
        import os

        os.environ["WANDB_PROJECT"] = "deepfake-detect"
        os.environ["WANDB_LOG_MODEL"] = "false"

    train()
