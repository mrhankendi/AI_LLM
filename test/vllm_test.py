from vllm import LLM, SamplingParams

# Create LLM instance using 4 GPUs
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=4,
    max_model_len=4096
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=50
)

prompts = [
    "Explain why the sky is blue in simple terms.",
    "Write a short poem about an A100 GPU."
]

outputs = llm.generate(prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Output: {output.outputs[0].text.strip()}\n")

