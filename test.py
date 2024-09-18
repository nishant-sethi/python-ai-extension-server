from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load the pre-trained T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')


# Example code snippet with a bug
code_to_analyze = "def multiply(a, b): return a + b"

# Prepare the input by adding an instruction prefix
input_text = f"find bugs: {code_to_analyze}"

# Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate the output from the model
outputs = model.generate(input_ids, max_length=50, temperature=0.7, do_sample=True,
                         num_return_sequences=1)

# Decode the output to readable text
suggested_correction = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Original Code: {code_to_analyze}")
print(f"Suggested Correction: {suggested_correction}")
