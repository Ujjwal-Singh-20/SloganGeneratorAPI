from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
import spacy
from functools import lru_cache

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request body
class SloganRequest(BaseModel):
    brand: str
    description: str
    industry: str
    tone: Optional[str] = "playful"
    num: Optional[int] = 5
    liked_slogan: Optional[str] = None

# Cached model loading
@lru_cache(maxsize=1)
def load_models():
    model = GPT2LMHeadModel.from_pretrained("./")
    tokenizer = GPT2Tokenizer.from_pretrained("./")
    return model, tokenizer

model, tokenizer = load_models()

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Tone presets
TONE_PRESETS = {
    "playful": {"temperature": 0.95, "top_p": 0.95, "repetition_penalty": 1.2},
    "bold": {"temperature": 0.8, "top_p": 0.9, "repetition_penalty": 1.45},
    "minimalist": {"temperature": 0.6, "top_p": 0.8, "repetition_penalty": 1.5},
    "luxury": {"temperature": 0.7, "top_p": 0.85, "repetition_penalty": 1.35},
    "classic": {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.25}
}

def summarize_description(text: str) -> str:
    """Extract key words from description using spaCy"""
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"]]
    return " ".join(keywords[:12])

@app.get("/")
def read_root():
    return {"message": "Welcome to Slogan Generator API. Use POST / to generate slogans."}

@app.post("/")
async def generate_slogans(request: SloganRequest):
    try:
        # Process description
        processed_desc = summarize_description(request.description)
        
        # Generate prompts based on presence of liked slogan
        if request.liked_slogan:
            prompt1 = (
                f"Create {request.industry} brand slogans similar to: '{request.liked_slogan}'\n"
                f"Brand: {request.brand}\n"
                f"Key Attributes: {processed_desc}\n"
                "Slogan:"
            )
            prompt2 = (
                f"Generate slogans in the style of: '{request.liked_slogan}'\n"
                f"For: {request.brand}\n"
                f"Details: {processed_desc}\n"
                "Slogan:"
            )
        else:
            prompt1 = (
                f"Create a {request.industry} brand slogan that's {request.tone} and unique.\n"
                f"Brand: {request.brand}\n"
                f"Attributes: {processed_desc}\n"
                "Slogan:"
            )
            prompt2 = (
                f"Write {request.tone} marketing slogans for this {request.industry} brand:\n"
                f"Name: {request.brand}\n"
                f"About: {processed_desc}\n"
                "Slogan:"
            )

        # Generation parameters
        gen_params = {
            **TONE_PRESETS[request.tone],
            "max_new_tokens": 25,
            "num_return_sequences": request.num,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id
        }

        # Generate from both prompts
        outputs1 = generator(prompt1, **gen_params)
        outputs2 = generator(prompt2, **gen_params)

        # Process and deduplicate slogans
        slogans = []
        for output_group in [outputs1, outputs2]:
            for o in output_group:
                raw = o['generated_text'].split("Slogan:")[-1].strip()
                clean = raw.split("\n")[0].replace('"', '').replace('(', '').split(".")[0].strip()
                if len(clean) > 4 and clean not in slogans:
                    slogans.append(clean)

        return {"slogans": slogans[:request.num * 2]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
