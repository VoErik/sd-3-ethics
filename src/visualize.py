import math
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_attention(model, tokenizer, prompt, model_name="CLIP", layer=-1):
    print(f"\nAnalyzing attention for prompt: '{prompt}' with {model_name}")

    tokens = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=tokenizer.model_max_length)
    input_ids = tokens.input_ids

    raw_token_strings = tokenizer.convert_ids_to_tokens(input_ids[0])

    actual_seq_len = tokens.attention_mask[0].sum().item()
    token_strings = [token for i, token in enumerate(raw_token_strings) if tokens.attention_mask[0][i] == 1]

    model.eval()
    with torch.no_grad():
        outputs = model(**tokens, output_attentions=True)

    attentions = outputs.attentions

    layer_attentions = attentions[layer]  # Shape: (1, num_heads, seq_len, seq_len) for batch_size=1
    avg_head_attentions = layer_attentions.mean(dim=1).squeeze(0) # Shape: (seq_len, seq_len)
    avg_head_attentions_np = avg_head_attentions.cpu().numpy()

    if "CLIP" in model_name: # assume CLIP models have <|startoftext|> at idx 0
        # exclude the startoftext token
        attention_matrix_to_plot = avg_head_attentions_np[1:actual_seq_len, 1:actual_seq_len]
        plot_token_strings = token_strings[1:actual_seq_len]
    else: # For T5, keep the original slicing
        attention_matrix_to_plot = avg_head_attentions_np[:actual_seq_len, :actual_seq_len]
        plot_token_strings = token_strings[:actual_seq_len]

    if "CLIP" in model_name:
        original_pooled_token_index = input_ids[0].argmax().item()
        if original_pooled_token_index > 0:
            pooled_token_index = original_pooled_token_index - 1
        else:
            pooled_token_index = actual_seq_len - 2
    else:
        pooled_token_index = actual_seq_len - 1

    if pooled_token_index >= len(plot_token_strings):
        print(f"Warning: Pooled token idx ({pooled_token_index}) is outside adjusted sequence length ({len(plot_token_strings)}). "
              f"Using last actual token.")
        pooled_token_index = len(plot_token_strings) - 1

    if not plot_token_strings:
        print("Warning: plot_token_strings is empty. Cannot determine pooled token representation or relevance scores.")
        return {
            "attention_matrix": None,
            "token_labels": [],
            "relevance_scores": {},
            "pooled_token_representation": "N/A",
            "prompt": prompt,
            "model_name": model_name
        }

    pooled_token_representation = plot_token_strings[pooled_token_index]

    attention_from_pooled_token = attention_matrix_to_plot[pooled_token_index, :]

    relevance_scores = {}
    for i, token_str in enumerate(plot_token_strings):
        score = attention_from_pooled_token[i]
        relevance_scores[token_str] = score

    return {
        "attention_matrix": attention_matrix_to_plot,
        "token_labels": plot_token_strings,
        "relevance_scores": relevance_scores,
        "pooled_token_representation": pooled_token_representation,
        "prompt": prompt,
        "model_name": model_name
    }

def plot_attention_heatmaps(all_attention_data, title="All", savepath=None):
    num_plots = len(all_attention_data)
    nrows = math.ceil(math.sqrt(num_plots))
    ncols = math.ceil(num_plots / nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 7), squeeze=False)
    axes = axes.flatten()

    for i, data in enumerate(all_attention_data):
        ax = axes[i]
        attention_matrix = data["attention_matrix"]
        token_labels = data["token_labels"]
        prompt = data["prompt"]
        model_name = data["model_name"]

        if attention_matrix is None or not token_labels:
            ax.set_title(f"No Attention Data for {model_name}\nPrompt: '{prompt}'")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        sns.heatmap(
            attention_matrix,
            xticklabels=token_labels,
            yticklabels=token_labels,
            cmap="viridis",
            annot=False,
            ax=ax
        )
        ax.set_xlabel("Attended-to Tokens (Key)")
        ax.set_ylabel("Attending Tokens (Query)")
        ax.set_title(f"Avg. Attention Heatmap - {model_name}\nPrompt: '{prompt}'")
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle(f"{title} Attention Heatmaps", y=1.02, fontsize=16) # Overall title
    if savepath:
        plt.savefig(f"{savepath}.png")
    else:
        plt.show()
    plt.close()

def plot_relevance_bar_charts(all_relevance_data, title="All", savepath=None):
    num_plots = len(all_relevance_data)
    nrows = math.ceil(math.sqrt(num_plots))
    ncols = math.ceil(num_plots / nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 5), squeeze=False)
    axes = axes.flatten()

    for i, data in enumerate(all_relevance_data):
        ax = axes[i]
        relevance_scores = data["relevance_scores"]
        pooled_token_representation = data["pooled_token_representation"]
        prompt = data["prompt"]
        model_name = data["model_name"]

        if not relevance_scores:
            ax.set_title(f"No Relevance Data for {model_name}\nPrompt: '{prompt}'")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        ax.bar(relevance_scores.keys(), relevance_scores.values(), color='skyblue')
        ax.set_xlabel("Token")
        ax.set_ylabel("Attention Score from Pooled Token")
        ax.set_title(f"Token Relevance from '{pooled_token_representation}' - {model_name}\nPrompt: '{prompt}'")
        ax.tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle(f"{title} Token Relevance Bar Charts", y=1.02, fontsize=16) # Overall title
    if savepath:
        plt.savefig(f"{savepath}.png")
    else:
        plt.show()
    plt.close()