from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-falcon-11b")
model = AutoModelForCausalLM.from_pretrained("./fine-tuned-falcon-11b")

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=500)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        response = generate_response(user_input)
        print(f"Bot: {response}")
