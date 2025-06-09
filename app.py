import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import spacy

# Load models
model = GPT2LMHeadModel.from_pretrained("./")
tokenizer = GPT2Tokenizer.from_pretrained("./")
tokenizer.pad_token = tokenizer.eos_token

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def summarize_description(text):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"]]
    return " ".join(keywords[:12])

def generate_slogans(brand, description, industry, tone="playful", num=5, liked_slogan=None):
    processed_desc = summarize_description(description)
    if liked_slogan:
        prompt1 = (
            f"Create {industry} brand slogans similar to: '{liked_slogan}'\n"
            f"Brand: {brand}\n"
            f"Key Attributes: {processed_desc}\n"
            "Slogan:"
        )
        prompt2 = (
            f"Generate slogans in the style of: '{liked_slogan}'\n"
            f"For: {brand}\n"
            f"Details: {processed_desc}\n"
            "Slogan:"
        )
    else:
        prompt1 = (
            f"Create a {industry} brand slogan that's {tone} and unique.\n"
            f"Brand: {brand}\n"
            f"Attributes: {processed_desc}\n"
            "Slogan:"
        )
        prompt2 = (
            f"Write {tone} marketing slogans for this {industry} brand:\n"
            f"Name: {brand}\n"
            f"About: {processed_desc}\n"
            "Slogan:"
        )

    tone_presets = {
        "playful": {"temperature": 0.95, "top_p": 0.95, "repetition_penalty": 1.2},
        "bold": {"temperature": 0.8, "top_p": 0.9, "repetition_penalty": 1.45},
        "minimalist": {"temperature": 0.6, "top_p": 0.8, "repetition_penalty": 1.5},
        "luxury": {"temperature": 0.7, "top_p": 0.85, "repetition_penalty": 1.35},
        "classic": {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.25}
    }

    gen_params = {
        **tone_presets[tone],
        "max_new_tokens": 25,
        "num_return_sequences": num,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    }

    # Generate from both prompts
    outputs1 = model.generate(**tokenizer(prompt1, return_tensors="pt"), **gen_params)
    outputs2 = model.generate(**tokenizer(prompt2, return_tensors="pt"), **gen_params)

    slogans = []
    for outputs in [outputs1, outputs2]:
        for output in outputs:
            raw = tokenizer.decode(output, skip_special_tokens=True)
            clean = raw.split("Slogan:")[-1].strip()
            clean = clean.split("\n")[0].replace('"', '').replace('(', '').split(".")[0].strip()
            if len(clean) > 4 and clean not in slogans:
                slogans.append(clean)

    return {"slogans": slogans[:num * 2]}

# Gradio interface
inputs = [
    gr.Textbox(label="Brand"),
    gr.Textbox(label="Description"),
    gr.Textbox(label="Industry"),
    gr.Dropdown(["playful", "bold", "minimalist", "luxury", "classic"], label="Tone", value="playful"),
    gr.Slider(1, 10, step=1, value=5, label="Number of Slogans"),
    gr.Textbox(label="Generate like this slogan (optional)", value=None)
]
outputs = gr.JSON(label="Generated Slogans")

interface = gr.Interface(
    fn=generate_slogans,
    inputs=inputs,
    outputs=outputs,
    title="Slogan Generator API",
    flagging_mode="never"
    # allow_flagging="never"
)

# Launch with API endpoint
interface.launch(show_api=True)
