from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

# Define system prompt and chat formatting
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, "
    "and put your final answer within \\boxed{{}} . The reasoning process and answer are enclosed within <think> </think> "
    "and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. "
    "Note that respond by English, NOT use other languages."
)

def process_single_prompt(question, tokenizer):
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )

# Existing function, no change
def compute_token_ranks(model, tokenizer, input_ids):
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
    
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    sorted_indices = shift_logits.argsort(dim=-1, descending=True)

    token_ranks = (sorted_indices == shift_labels.unsqueeze(-1)).nonzero()
    ranks = torch.zeros_like(shift_labels)
    for b, pos, rank in token_ranks:
        ranks[b, pos] = rank + 1

    return ranks.cpu().tolist()[0]

# Main analysis logic
def analyze_rank_shift(base_model_name, tuned_model_name, question, output):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tuned_tokenizer = AutoTokenizer.from_pretrained(tuned_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device).eval()
    tuned_model = AutoModelForCausalLM.from_pretrained(tuned_model_name).to(device).eval()

    assert base_tokenizer.get_vocab() == tuned_tokenizer.get_vocab(), "Tokenizers must match."
    tokenizer = base_tokenizer

    # Use your chat-style prompt
    formatted_prompt = process_single_prompt(question, tokenizer)
    full_text = formatted_prompt + output

    # Tokenize
    input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
    input_len = prompt_ids.size(1)
    output_len = input_ids.size(1) - input_len

    # Compute ranks
    base_ranks = compute_token_ranks(base_model, tokenizer, input_ids)
    tuned_ranks = compute_token_ranks(tuned_model, tokenizer, input_ids)

    # Focus on output token positions
    base_ranks_output = base_ranks[input_len - 1:]
    tuned_ranks_output = tuned_ranks[input_len - 1:]
    rank_shifts = [base - tuned for base, tuned in zip(base_ranks_output, tuned_ranks_output)]

    output_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][input_len:])
    print(f"\n{'Token':<20} | {'Base Rank':<10} | {'Tuned Rank':<11} | {'Rank Shift'}")
    print("-" * 60)
    for tok, base_r, tuned_r, shift in zip(output_tokens, base_ranks_output, tuned_ranks_output, rank_shifts):
        print(f"{repr(tok):<20} | {base_r:<10} | {tuned_r:<11} | {shift}")

    return rank_shifts

# Example execution
if __name__ == "__main__":
    base_model = "/data/shuozhe/saved_model/Qwen2.5-0.5B"
    tuned_model = "/data/shuozhe/saved_model/Qwen2.5-0.5B-Instruct"
    question = "What is the sum of 3 and 5?"
    output = "<think>To solve 3 + 5, just add the two numbers.</think> <answer>\\boxed{8}</answer>"

    analyze_rank_shift(base_model, tuned_model, question, output)
