from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
from peft import PeftModel, PeftConfig

model_path = "output/vast6000/checkpoint-32904"

peft_config = PeftConfig.from_pretrained(model_path)

base_model_name_or_path = peft_config.base_model_name_or_path

# default: Load the model on the available device(s)
model = AutoModelForImageTextToText.from_pretrained(base_model_name_or_path, dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(
    model,
    model_path,
    device_map="auto",
    torch_dtype="auto",
)

model = model.merge_and_unload()


# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = AutoModelForImageTextToText.from_pretrained(
#     "Qwen/Qwen3-VL-235B-A22B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained(model_path)

import os.path as osp

save_path = "output/merged_model"
save_path = osp.join(save_path, model_path.replace("output/", ""))

model.save_pretrained(save_path)
processor.save_pretrained(save_path)

print(f"Merged model saved to {save_path}")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "data/image copy.png",
            },
            {"type": "text", "text": "Is this image real or fake? Answer with one word: Real or Fake."},
            # {"type": "text", "text": "Explain the image in detail."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output_text)
