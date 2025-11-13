from transformers import AutoModelForImageTextToText, AutoProcessor

# default: Load the model on the available device(s)
model_name = "Qwen/Qwen3-VL-2B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(model_name, dtype="auto", device_map="auto")

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = AutoModelForImageTextToText.from_pretrained(
#     "Qwen/Qwen3-VL-235B-A22B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

import torchvision

video_path = "data/sample_video_3.mp4"
reader = torchvision.io.VideoReader(video_path, "video")
# reader.seek(2.0)
# frame = next(reader)
frames = []
for frame in reader.seek(0.0):
    frames.append(frame["data"])

print(reader.get_metadata())
fps = reader.get_metadata()["video"]["fps"]

print(len(frames), "frames loaded.")

processor = AutoProcessor.from_pretrained(model_name)

from PIL import Image

img_path = "data/sample_image_2.png"
img = Image.open(img_path).convert("RGB")
print(img)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": img,
            },
            {"type": "text", "text": "Is this image real or fake? Answer with one word: Real or Fake."},
        ],
    }
]

# Preparation for inference
# inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", video_metadata={"fps": 24, "total_num_frames": len(frames)})
# inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inputs = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_dict=True)
print(inputs)
inputs = processor(inputs, return_tensors="pt")
print(inputs)
# print("Inputs:", inputs)
# print("Decoded input_ids:", processor.batch_decode(inputs.input_ids, skip_special_tokens=False))
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output_text)
