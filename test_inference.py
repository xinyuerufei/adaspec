"""
Simple inference script to test Qwen2.5-0.5B on a math problem.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Math problem
    question = """In a graveyard, there are 20 skeletons.  Half of these skeletons are adult women, and the remaining number are split evenly between adult men and children.  If an adult woman has 20 bones in their body, and a male has 5 more than this, and a child has half as many as an adult woman, how many bones are in the graveyard?
Answer the above question. First think step by step and then answer the final number."""

    print("\n" + "="*60)
    print("Question:")
    print(question)
    print("="*60)
    
    # Tokenize input
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    
    # Generate with parameters similar to run_train.sh
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.2,
            top_k=40,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and print result
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove the input prompt)
    response = generated_text[len(question):]
    
    print("\nGenerated Response:")
    print("-"*60)
    print(response)
    print("-"*60)
    
    print("\nFull Output:")
    print("="*60)
    print(generated_text)
    print("="*60)


if __name__ == "__main__":
    main()

