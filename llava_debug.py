from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
from xtuner.model.modeling_llava import CustomLlavaForConditionalGeneration
from xtuner.model.modeling_llama import CustomLlamaForCausalLM
from transformers import LlamaConfig

# Load the model and processor
model = CustomLlavaForConditionalGeneration.from_pretrained(pretrained_model_name_or_path="/data/kesun/work_dirs/llava_v15_13b_finetune_AIGC_lora_best1128/iter_10000_hf")
model_name = "/data/kesun/work_dirs/llava_v15_13b_finetune_AIGC_lora_best1128/iter_10000_merged"  # 替换为你的预训练模型名称
# tokenizer = LlamaTokenizer.from_pretrained(model_name)
# config = LlamaConfig.from_pretrained(model_name)
language_model = CustomLlamaForCausalLM.from_pretrained(model_name)
model.language_model = language_model
processor = AutoProcessor.from_pretrained("/data/kesun/work_dirs/llava_v15_13b_finetune_AIGC_lora_best1128/iter_10000_hf")

# Prepare the input prompt and image
prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
image_file = './DeepStack-VL/assets/logo.png'

# url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(image_file)

# Process the input
inputs = processor(images=image, text=prompt, return_tensors="pt")

# Forward pass through the model
outputs = model(**inputs)

# Extract the logits from the outputs
logits = outputs.logits

# Generate text based on the logits
# generate_ids = model.generate(inputs["input_ids"], max_new_tokens=15)
# generated_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# print(generated_text)