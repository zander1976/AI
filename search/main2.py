# test_hello.py
import time

from mlx_lm import generate, load
from rich import print

print()
print("[bold green]Starting MLX LLaMA test...[/bold green]")
print()

# -----------------------------
# 0️⃣ Load MLX model (cached if already downloaded)
# -----------------------------
print("[blue]Step 1: Loading MLX model...[/blue]")
print()
start_time = time.time()

# Change to your desired model
model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"
model, tokenizer = load(model_name)

print(f"[green]Model loaded (took {time.time() - start_time:.2f}s)[/green]")
print()

# -----------------------------
# 1️⃣ Prepare prompt
# -----------------------------
print()
prompt = "Hello, how are you?"

# Apply chat template if available
if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

# -----------------------------
# 2️⃣ Generate response
# -----------------------------
output = generate(model, tokenizer, prompt=prompt, max_tokens=32, verbose=True)
print("[bold green]Output:[/bold green]", output)
print()
