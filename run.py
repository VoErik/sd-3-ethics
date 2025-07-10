from src.sd_pipe import run_pipeline_from_csv, run_pipeline_single, SDConfig, MODEL_DICT
from src.visualize import analyze_attention, plot_attention_heatmaps, plot_relevance_bar_charts
from dotenv import load_dotenv
from huggingface_hub import login

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["g", "v"], default="g", help="g enerate, v isualize")
parser.add_argument("--num_imgs", type=int, default=10)
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--guidance", type=float, default=7.0)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--offload_to_cpu", type=bool, default=True)
parser.add_argument("--quantized", type=bool, default=False)
parser.add_argument("--csv_path", type=str, default=None, help="When provided will use prompts from csv.")
parser.add_argument("--model", type=str, default="turbo-3.5")
parser.add_argument("--prompts", type=str, nargs='*', default=None, help="Only for visualization.")
parser.add_argument("--layer", type=int, default=-1, help="Layer to visualize.")


if __name__ == "__main__":
    args = parser.parse_args()
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)

    if args.task == "g":
        cfg = SDConfig(
            model_id=args.model,
            width=args.width,
            height=args.height,
            inference_steps=args.steps,
            enable_model_cpu_offload=args.offload_to_cpu,
            use_quantized=args.quantized,
            guidance_scale=args.guidance,
            csv_path=args.csv_path,
            num_images_per_prompt=args.num_imgs,
        )
        if args.csv_path is not None:
            run_pipeline_from_csv(cfg)
        else:
            run_pipeline_single(cfg)

    elif args.task == "v":
        from transformers.models import CLIPTextModelWithProjection, T5EncoderModel
        from transformers.models import T5TokenizerFast, CLIPTokenizer

        savedir = "./attention_maps"
        os.makedirs(savedir, exist_ok=True)
        model_id = MODEL_DICT[args.model]
        text_enc_1 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder='text_encoder')
        text_enc_2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder='text_encoder_2')
        text_enc_3 = T5EncoderModel.from_pretrained(model_id, subfolder='text_encoder_3')
        tok_1 = CLIPTokenizer.from_pretrained(model_id, subfolder='tokenizer')
        tok_2 = CLIPTokenizer.from_pretrained(model_id, subfolder='tokenizer_2')
        tok_3 = T5TokenizerFast.from_pretrained(model_id, subfolder='tokenizer_3')

        prompt1 = args.prompts[0] if args.prompts else str(input("Enter 1st prompt: "))
        prompt2 = args.prompts[1] if args.prompts else str(input("Enter 2nd prompt: "))

        all_attention_maps_data = []
        all_relevance_charts_data = []

        print("\n--- Analyzing Prompt 1 with text_enc_1 ---")
        data = analyze_attention(
            text_enc_1, tok_1, prompt1, model_name="Text Encoder 1 (CLIP)", layer=args.layer
        )
        all_attention_maps_data.append(data)
        all_relevance_charts_data.append(data)

        print("\n--- Analyzing Prompt 2 with text_enc_1 ---")
        data = analyze_attention(
            text_enc_1, tok_1, prompt2, model_name="Text Encoder 1 (CLIP)", layer=args.layer
        )
        all_attention_maps_data.append(data)
        all_relevance_charts_data.append(data)

        print("\n--- Analyzing Prompt 1 with text_enc_2 ---")
        data = analyze_attention(
            text_enc_2, tok_2, prompt1, model_name="Text Encoder 2 (CLIP ViT)", layer=args.layer
        )
        all_attention_maps_data.append(data)
        all_relevance_charts_data.append(data)

        print("\n--- Analyzing Prompt 2 with text_enc_2 ---")
        data = analyze_attention(
            text_enc_2, tok_2, prompt2, model_name="Text Encoder 2 (CLIP ViT)", layer=args.layer
        )
        all_attention_maps_data.append(data)
        all_relevance_charts_data.append(data)

        print("\n--- Analyzing Prompt 1 with text_enc_3 (T5) ---")
        data = analyze_attention(
            text_enc_3, tok_3, prompt1, model_name="Text Encoder 3 (T5)", layer=args.layer
        )
        all_attention_maps_data.append(data)
        all_relevance_charts_data.append(data)

        print("\n--- Analyzing Prompt 2 with text_enc_3 (T5) ---")
        data = analyze_attention(
            text_enc_3, tok_3, prompt2, model_name="Text Encoder 3 (T5)", layer=args.layer
        )
        all_attention_maps_data.append(data)
        all_relevance_charts_data.append(data)

        print("\n--- Generating Plots ---")
        plot_attention_heatmaps(
            all_attention_maps_data, savepath=os.path.join(savedir, "attention_heatmaps")
        )
        plot_relevance_bar_charts(
            all_relevance_charts_data, savepath=os.path.join(savedir, "relevance_bar_charts")
        )
    else:
        raise ValueError("Task must be 'generate' or 'visualize'")