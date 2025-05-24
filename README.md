# Qwen2.5-Math-1.5B Local Machine Implementation

This project demonstrates how to run the [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) model locally for mathematical reasoning and language generation using Hugging Face Transformers.


## Requirements

- Python 3.8+
- PyTorch (with GPU support for better performance)
- Transformers
- Accelerate (optional, for device mapping)

## Install dependencies:

     pip install torch transformers accelerate

## How to Run

**1. Download the model locally:**

Download the model files from Hugging Face and place them in a directory, e.g., /path/to/Qwen2.5-Math-1.5B.

**2. Edit the notebook:**

Update the path to the downloaded model:

    tokenizer = AutoTokenizer.from_pretrained('/path/to/Qwen2.5-Math-1.5B')
    model = AutoModelForCausalLM.from_pretrained('/path/to/Qwen2.5-Math-1.5B', torch_dtype=torch.bfloat16)

**3. Run the notebook:**
Launch Jupyter Notebook and execute each cell to:

- Tokenize an input prompt

- Perform inference

- Decode the model output

Example:

    input_text = "Solve: 123 + 456 = ?"
    inputs = tokenizer(input_text, return_tensors='pt')
    outputs = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0]))

## Model Details
- Model: Qwen2.5-Math-1.5B

- Type: Causal Language Model

- Use Case: Advanced reasoning and mathematical problem solving

- Framework: Hugging Face Transformers

## Notes
- Make sure your device supports `bfloat16` or change `torch_dtype=torch.float16` or `float32` as needed.

- Consider using `model.to("cuda")` for faster inference on GPU.

## Reference
- [Qwen2.5-Math-1.5B on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)

- [Transformers Documentation](https://huggingface.co/docs/transformers)


