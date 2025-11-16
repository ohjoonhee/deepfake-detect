MAX_SAMPLE_IN_DATASET = 3_000_000  # 3 million samples

from dataclasses import dataclass, field
from typing import Optional, List, Any, Union

from datasets import load_dataset
from transformers import AutoProcessor, AutoConfig
from trl import SFTTrainer, SFTConfig, TrlParser, ModelConfig
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

# Need for custom data collator
# from trl.data_utils import is_conversational, apply_chat_template, prepare_multimodal_messages
from trl.data_utils import is_conversational, apply_chat_template
from trl.trainer.utils import flush_left

from qwen_vl_utils import process_vision_info
import torch

from transformers.video_utils import load_video

from itertools import takewhile

import random
import cv2
from PIL import Image
import io

import datasets

from transformers.integrations.integration_utils import is_wandb_available

import numpy as np


import evaluate


def prepare_multimodal_messages(messages: list[dict[str, Any]], images: List, video: List) -> None:
    """
    Convert messages into a structured multimodal format if needed.

    Each message's content is transformed from a raw string into a list of typed parts. The first user message is
    prefixed with an image placeholder, while all other user and assistant messages are wrapped as text entries.

    Args:
        messages (`list[dict[str, Any]]`):
            Messages with `"role"` and `"content"`. Content may be a raw string before transformation.
        num_images (`int`):
            Number of images to include in the first user message. This is used to determine how many image
            placeholders to add.

    Example:
    ```python
    # Input
    [
        {"role": "user", "content": "What's in this image?"},
        {"role": "assistant", "content": "It looks like a cat."},
    ]

    # Output (num_images=1)
    [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What's in this image?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "It looks like a cat."}]},
    ]
    ```
    """
    image_included = False
    for message in messages:
        if message["role"] == "system":
            if isinstance(message["content"], str):  # if already prepared, the content will be a list
                message["content"] = [{"type": "text", "text": message["content"]}]
        elif message["role"] == "user":
            if isinstance(message["content"], str) and not image_included:
                # placeholders = [{"type": "image"}] * num_images
                placeholders = []
                for img in images:
                    placeholders.append({"type": "image", "image": img})
                if video is not None:
                    # Add video to the messages with proper video data
                    frames = video.get_frames_played_in_range(0.0, 5.0)  # TODO: parameterize

                    video_obj, _ = load_video(frames.data, fps=video.metadata.average_fps_from_header)
                    placeholders.append({"type": "video", "video": video_obj})
                message["content"] = [*placeholders, {"type": "text", "text": message["content"]}]
                image_included = True
            elif isinstance(message["content"], str) and image_included:
                message["content"] = [{"type": "text", "text": message["content"]}]
        elif message["role"] == "assistant":
            if isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"]}]
        else:
            raise ValueError(f"Invalid role in message: {message['role']}. Expected 'user', 'assistant', or 'system'.")


class MyDataCollator(DataCollatorForVisionLanguageModeling):
    def _collate_prompt_completion(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if self.pad_to_multiple_of is not None:
            raise NotImplementedError("Padding to a multiple of a value is not yet implemented for vision-language modeling and " "prompt-completion data yet.")
        images = [example["images"] for example in examples]
        videos = [example["video"] for example in examples] if "video" in examples[0] else None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if all(img_list == [] for img_list in images):
            images = None
        if is_conversational(examples[0]):  # conversational case
            for example in examples:
                images = example["images"] if example["images"] != None else []  # All datasets even video datasets should have images key for vision_dataset=True
                video = example.get("video", None)
                prepare_multimodal_messages(example["prompt"] + example["completion"], images, video)

        processed_prompts = self.processor.apply_chat_template(
            [example["prompt"] for example in examples],
            tokenize=True,
            add_generation_prompt=True,
            continue_final_message=False,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            # add_special_tokens=False,  # to avoid adding the BOS, twice see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        )

        untokenized_prompts = self.processor.apply_chat_template(
            [example["prompt"] for example in examples],
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False,
        )

        # Process completions
        completions = []
        for example, untokenized_prompt in zip(examples, untokenized_prompts):
            prompt_completion = self.processor.apply_chat_template(
                example["prompt"] + example["completion"],
                # tools=tools,
                tokenize=False,
                **example.get("chat_template_kwargs", {}),
                # **template_kwargs,
            )

            completion = prompt_completion[len(untokenized_prompt) :]
            completions.append(completion)

        processed_completions = self.processor(
            text=completions,
            padding=True,
            padding_side="right",
            return_tensors=self.return_tensors,
            add_special_tokens=False,  # to avoid adding the BOS, twice see https://huggingface.co/blog/qgallouedec/gotchas-in-tokenizer-behavior#7-chat-template-and-tokenization-dont-compose-due-to-special-tokens
        )

        # Concatenate prompts and completions
        prompt_ids, completion_ids = processed_prompts["input_ids"], processed_completions["input_ids"]
        prompt_mask, completion_mask = processed_prompts["attention_mask"], processed_completions["attention_mask"]
        input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        attention_mask = torch.cat((prompt_mask, completion_mask), dim=1)
        completion_mask = torch.cat((torch.zeros_like(prompt_mask), completion_mask), dim=1)
        if "token_type_ids" in processed_prompts:  # special case for Gemma
            prompt_token_type_ids = processed_prompts["token_type_ids"]
            completion_token_type_ids = processed_completions["token_type_ids"]
            token_type_ids = torch.cat((prompt_token_type_ids, completion_token_type_ids), dim=1)

        # Flush left to reduce padding
        if "token_type_ids" in processed_prompts:
            attention_mask, input_ids, completion_mask, token_type_ids = flush_left(attention_mask, input_ids, completion_mask, token_type_ids)
        else:
            attention_mask, input_ids, completion_mask = flush_left(attention_mask, input_ids, completion_mask)

        # Truncate if necessary
        if self.max_length is not None:
            input_ids = input_ids[:, : self.max_length]
            attention_mask = attention_mask[:, : self.max_length]
            completion_mask = completion_mask[:, : self.max_length]
            if "token_type_ids" in processed_prompts:
                token_type_ids = token_type_ids[:, : self.max_length]

        # Create labels and mask padding tokens
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        if self.completion_only_loss:
            labels[completion_mask == 0] = -100

        # Build the output dictionary
        output = processed_prompts  # we take processed_prompts because it contains the images
        output["input_ids"] = input_ids
        output["attention_mask"] = attention_mask
        output["labels"] = labels
        if "token_type_ids" in processed_prompts:
            output["token_type_ids"] = token_type_ids

        # Save output to disk for debugging
        # torch.save(output, "data_collator_output.pt")

        # import sys

        # sys.exit(0)

        return output


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
    # train_eval_split: Optional[float] = field(
    #     default=None,
    #     metadata={"help": "If specified, a portion of the training set will be set aside for evaluation."},
    # )
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
    num_proc_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of subprocesses to use for data processing. 0 means that the data will be loaded in the main process."},
    )
    degrade_fake: Optional[list[bool]] = field(
        default=None,
        metadata={"help": "A list of degradation types to apply to the images."},
    )
    num_total_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Total number of samples to use from the dataset(s)."},
    )


def estimate_blur_laplacian(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def degrade_image_to_match_laion5(img_pil, real_blur_vals, real_res_vals, noise_var=0.0005, jpeg_quality_range=(70, 95)):
    """
    Lightly degrades an image to mimic real training images' resolution and blur distribution.
    """
    # if seed is not None:
    #     random.seed(seed)
    #     np.random.seed(seed)

    # === Step 1: Resize to match real training image resolution ===
    if random.random() < 0.2:
        target_h, target_w = random.choice(real_res_vals)
        orig_w, orig_h = img_pil.size
        orig_area = orig_w * orig_h
        target_area = target_h * target_w
        scale = (target_area / orig_area) ** 0.5
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))

        img_pil = img_pil.resize((new_w, new_h), Image.BILINEAR)
    img_np = np.array(img_pil)

    if random.random() < 0.2:
        target_blur = np.random.choice(real_blur_vals)
        blur_val = estimate_blur_laplacian(img_np)
        if blur_val > target_blur * 1.2:
            # GaussianBlur in OpenCV is highly optimized and releases the GIL
            img_np = cv2.GaussianBlur(img_np, (0, 0), sigmaX=0.3, sigmaY=0.3)

    if random.random() < 0.2:
        # === Step 3: Add light Gaussian noise (OpenCV) ===
        sigma = int(255 * (noise_var**0.5))
        if sigma > 0:
            noise = np.zeros_like(img_np, dtype=np.int16)
            cv2.randn(noise, 0, sigma)  # in‑place Gaussian noise
            img_np = cv2.add(img_np.astype(np.int16), noise, dtype=cv2.CV_8U)

    if random.random() < 0.2:
        # === Step 4: Mild JPEG compression ===
        quality = np.random.randint(*jpeg_quality_range)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode(".jpg", img_np, encode_param)
        img_np = cv2.imdecode(encimg, cv2.IMREAD_COLOR)

    return Image.fromarray(img_np)


# def preprocess(example):
#     label_map = {0: "Fake", 1: "Real"}
#     label = label_map[example["label"]]

#     prompt = [
#         {
#             "role": "user",
#             "content": "Is this image real or fake? Answer with one word: Real or Fake.",
#         },
#     ]
#     completion = [
#         {"role": "assistant", "content": f"{label}"},
#     ]

#     return {"prompt": prompt, "completion": completion, "images": [example["image"]]}


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

    # Quantization args
    if model_args.load_in_4bit or model_args.load_in_8bit:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            use_bnb_nested_quant=model_args.use_bnb_nested_quant,
        )

    else:
        bnb_config = None

    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        dtype=model_args.dtype,
        attn_implementation=model_args.attn_implementation,
        quantization_config=bnb_config,
    )

    # Peft config
    if model_args.use_peft:
        from peft import get_peft_model, LoraConfig

        if bnb_config is not None:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)  # TODO: fix hardcoded False

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
    # If dataset name is string, convert to list for uniform processing
    if isinstance(script_args.dataset_name, str) or script_args.dataset_name is None:
        script_args.dataset_name = [script_args.dataset_name]

        # Assert that train_eval_split is only used when a single dataset is specified
        if script_args.dataset_config is not None:
            script_args.dataset_config = [script_args.dataset_config]

        if script_args.train_split is not None:
            script_args.train_split = [script_args.train_split]

    train_datasets = []
    eval_datasets = []

    signature_features = datasets.Features(
        {
            "prompt": datasets.List({"role": datasets.Value("string"), "content": datasets.Value("string")}),
            "completion": datasets.List({"role": datasets.Value("string"), "content": datasets.Value("string")}),
            "images": datasets.List(datasets.Image()),
            "video": datasets.Video(),
            "label": datasets.ClassLabel(num_classes=2, names=["Real", "Fake"]),
            "degrade_fake": datasets.Value("bool"),
        }
    )
    signature_columns = list(signature_features.keys())

    for name, config, train_split, eval_split, degrade in zip(
        script_args.dataset_name,
        script_args.dataset_config or [None] * len(script_args.dataset_name),
        script_args.train_split or ["train"] * len(script_args.dataset_name),
        script_args.eval_split or [None] * len(script_args.dataset_name),
        script_args.degrade_fake or [True] * len(script_args.dataset_name),
    ):
        print("Dataset name:", name)

        train_dataset = load_dataset(name, config, split=train_split, streaming=True)

        print("Original train dataset features:", train_dataset.features.keys())

        column_names = train_dataset.column_names

        if "images" not in column_names and "image" in column_names:
            # train_dataset = train_dataset.rename_column("image", "images")
            def rename_image_column(example):
                example["images"] = [example["image"]]  # wrap single image into list
                return example

            train_dataset = train_dataset.map(rename_image_column, remove_columns=["image"])

        if "prompt" not in column_names:
            train_dataset = train_dataset.add_column("prompt", [None] * MAX_SAMPLE_IN_DATASET)
        if "completion" not in column_names:
            train_dataset = train_dataset.add_column("completion", [None] * MAX_SAMPLE_IN_DATASET)
        if "video" not in column_names:
            train_dataset = train_dataset.add_column("video", [None] * MAX_SAMPLE_IN_DATASET)
        if "images" not in column_names and "image" not in column_names:
            train_dataset = train_dataset.add_column("images", [None] * MAX_SAMPLE_IN_DATASET)

        train_dataset = train_dataset.remove_columns([col for col in column_names if col not in signature_columns])

        train_dataset = train_dataset.add_column("degrade_fake", [degrade] * MAX_SAMPLE_IN_DATASET)

        train_dataset = train_dataset.cast(signature_features)

        if eval_split is not None:
            if isinstance(eval_split, float):
                # if train_dataset ["label"] is datasets.ClassLabel, we can do stratified split
                raise ValueError("train_eval_split is not supported with streaming datasets.")
            elif isinstance(eval_split, str):
                eval_dataset = load_dataset(name, config, split=eval_split, streaming=True)

                eval_column_names = eval_dataset.column_names
                if "images" not in eval_column_names and "image" in eval_column_names:
                    # eval_dataset = eval_dataset.rename_column("image", "images")
                    def rename_image_column_eval(example):
                        example["images"] = [example["image"]]  # wrap single image into list
                        return example

                    eval_dataset = eval_dataset.map(rename_image_column_eval, remove_columns=["image"])
                if "prompt" not in eval_column_names:
                    eval_dataset = eval_dataset.add_column("prompt", [None] * MAX_SAMPLE_IN_DATASET)
                if "completion" not in eval_column_names:
                    eval_dataset = eval_dataset.add_column("completion", [None] * MAX_SAMPLE_IN_DATASET)
                if "video" not in eval_column_names:
                    eval_dataset = eval_dataset.add_column("video", [None] * MAX_SAMPLE_IN_DATASET)
                if "images" not in eval_column_names and "image" not in eval_column_names:
                    eval_dataset = eval_dataset.add_column("images", [None] * MAX_SAMPLE_IN_DATASET)

                eval_dataset = eval_dataset.remove_columns([col for col in eval_column_names if col not in signature_columns])
                eval_dataset = eval_dataset.add_column("degrade_fake", [degrade] * MAX_SAMPLE_IN_DATASET)

                eval_dataset = eval_dataset.cast(signature_features)

            else:
                raise ValueError("eval_split must be either float or str")
        else:
            eval_dataset = None

        print(f"Train dataset {name}:", train_dataset)
        print(f"Eval dataset {name}:", eval_dataset)

        train_datasets.append(train_dataset)
        eval_datasets.append(eval_dataset)

    train_dataset = datasets.interleave_datasets(train_datasets)  

    eval_datasets = [ds for ds in eval_datasets if ds is not None]
    eval_dataset = datasets.interleave_datasets(eval_datasets) if len(eval_datasets) > 0 else None

    # Apply degradation to fake images in training set
    real_train_stats = np.load("asset/real_train_stats.npz")
    real_blur_vals = real_train_stats["blur_vals"]
    real_res_vals = real_train_stats["res_vals"]

    def preprocess(example):
        is_image = example["images"] is not None and len(example["images"]) > 0
        if is_image:
            imgs = []
            for img in example["images"]:
                if example["degrade_fake"] and example["label"] == 1:
                    img = degrade_image_to_match_laion5(img, real_blur_vals, real_res_vals)
                imgs.append(img)
            example["images"] = imgs
        else:  # Video case
            pass

        question_type = "image" if is_image else "video"

        # Map string labels to integer: 0=real, 1=fake
        if example["prompt"] is None:
            example["prompt"] = [
                {
                    "role": "user",
                    "content": f"Is this {question_type} real or fake? Answer with one word: Real or Fake.",
                },
            ]
        if example["completion"] is None:
            example["completion"] = [
                {
                    "role": "assistant",
                    "content": "Fake" if example["label"] == 1 else "Real",
                }
            ]

        return example

    train_dataset = train_dataset.map(
        preprocess,
        # fn_kwargs={"processor": processor},
        remove_columns=[col for col in train_dataset.column_names if col not in signature_columns],
    )
    eval_dataset = (
        eval_dataset.map(
            preprocess,
            # fn_kwargs={"processor": processor},
            remove_columns=[col for col in eval_dataset.column_names if col not in signature_columns],
        )
        if eval_dataset is not None
        else None
    )

    print("Final train dataset:\n", train_dataset)
    print("Final eval dataset:\n", eval_dataset)

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

            # Decode generated summaries into text
            decoded_preds = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)

            # Replace -100 in the labels as we can't decode them
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            # Decode reference summaries into text
            decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
            # ROUGE expects a newline after each sentence
            # print("===== After Decoding =====")
            # print([len(i) for i in inputs])
            # print(decoded_preds)
            # decoded_preds = [pred.strip()[len(i) :] for (pred, i) in zip(decoded_preds, inputs)]
            decoded_preds = [pred.strip() for (pred) in decoded_preds]

            decoded_labels = [label.strip() for label in decoded_labels]

            aggregator.add_batch(decoded_preds, decoded_labels)

            # Debug prints
            # print("Decoded preds:", decoded_preds)
            # print("Decoded labels:", decoded_labels)

            if compute_result:
                return aggregator.compute()

    else:
        compute_metrics = None

    # Decide whether to use completion-only loss: if not specified, then it is set to True if the dataset format
    # is prompt-completion, and False if the dataset format is language modeling.
    dataset_sample = next(iter(train_dataset))
    if training_args.completion_only_loss is None:
        completion_only_loss = "prompt" in dataset_sample and "completion" in dataset_sample
    else:
        completion_only_loss = training_args.completion_only_loss

    data_collator = MyDataCollator(
        processor=processor,
        max_length=training_args.max_length,
        completion_only_loss=completion_only_loss,
        pad_to_multiple_of=training_args.pad_to_multiple_of,
        dataset_text_field=training_args.dataset_text_field,
    )

    # if training_args.dataset_kwargs is None:
    #     training_args.dataset_kwargs = {}
    # training_args.dataset_kwargs["skip_prepare_dataset"] = True  # We have already prepared the dataset

    try:
        num_devices = max(1, torch.cuda.device_count())
        print("Number of devices:", num_devices)
    except ImportError:
        num_devices = 1
        print("Getting number of devices failed, defaulting to 1.")
    
    # Calculate max_steps based on num_total_samples
    max_steps = (script_args.num_total_samples // (training_args.per_device_train_batch_size * num_devices)) // training_args.gradient_accumulation_steps * training_args.num_train_epochs
    training_args.max_steps = int(max_steps)


    training_args.remove_unused_columns = False  # To avoid removing images column needed by the data collator
    training_args.max_length = None  # We handle max_length in the data collator

    print("========= Training starting... ========")

    # 4. Create Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # 5. Train
    trainer.train()

    # 6. Save
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    import sys

    sys.argv += ["--config", "configs/streaming_gravex_openfake.yaml"]

    if is_wandb_available():
        import os

        os.environ["WANDB_PROJECT"] = "deepfake-detect"
        os.environ["WANDB_LOG_MODEL"] = "false"

    train()
