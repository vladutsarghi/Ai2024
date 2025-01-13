# import requests
# import json
#
# url = "http://localhost:8000/v1/chat/completions"
#
# data = {
#     "model": "meta-llama/Llama-3.3-70B-Instruct",
#     "messages": [
#         {
#             "role": "user",
#             "content": "What is the capital of France?"
#         }
#     ]
# }
#
# response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(data))
#
# if response.status_code == 200:
#     print("Response:", response.json())
# else:
#     print("Error:", response.status_code, response.text)
#

# import torch
# from transformers import pipeline
#
# pipe = pipeline(
#     "text-generation",
#     model="google/gemma-2-2b-it",
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     # device="cuda",  # replace with "mps" to run on a Mac device
# )
#
# messages = [
#     {"role": "user", "content": "Who are you? Please, answer in pirate-speak."},
# ]
#
# outputs = pipe(messages, max_new_tokens=256)
# assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
# print(assistant_response)
# # Ahoy, matey! I be Gemma, a digital scallywag, a language-slingin' parrot of the digital seas. I be here to help ye with yer wordy woes, answer yer questions, and spin ye yarns of the digital world.  So, what be yer pleasure, eh? 🦜



from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Interactive input loop
while True:
    input_text = input("Enter your text (or 'exit' to quit): ")
    if input_text.lower() == 'exit':
        break

    # Process input and generate output
    input_ids = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**input_ids, max_new_tokens=32)

    # Decode and print the response
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
