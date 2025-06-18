import torch
from diffusers import StableDiffusionPipeline
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Load model with LoRA
base_model = "runwayml/stable-diffusion-v1-5"
lora_repo = "nexo-here/shohan-lora"
lora_filename = "shohan-lora.safetensors"

# Make sure to login with `huggingface-cli login` or set token via environment
hf_token = os.getenv("HUGGINGFACE_TOKEN")

pipe = StableDiffusionPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    use_auth_token=hf_token
).to("cuda" if torch.cuda.is_available() else "cpu")

# Load the LoRA weights from HuggingFace
pipe.load_lora_weights(
    pretrained_model_name_or_path=lora_repo,
    weight_name=lora_filename,
    use_auth_token=hf_token
)

pipe.safety_checker = None  # Optional: disable NSFW filter
print("✅ LoRA Model loaded successfully!")

@app.route("/gen", methods=["GET"])
def generate():
    prompt = request.args.get("prompt", "A portrait of Shohan in studio lighting")
    try:
        image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({
            "prompt": prompt,
            "image": f"data:image/png;base64,{img_str}"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
