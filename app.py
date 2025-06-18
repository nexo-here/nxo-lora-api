import os
import io
import torch
from flask import Flask, request, jsonify
from PIL import Image
from diffusers import StableDiffusionPipeline

app = Flask(__name__)

# HuggingFace Token (Set in Render Secrets)
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# Base Model and LoRA Info
base_model = "runwayml/stable-diffusion-v1-5"
lora_repo = "nexo-here/shohan-lora"
lora_filename = "shohan-lora.safetensors"

# Load the base pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    use_auth_token=hf_token
)

# Move model to device
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Load LoRA weights from HuggingFace
pipe.load_lora_weights(
    pretrained_model_name_or_path=lora_repo,
    weight_name=lora_filename,
    use_auth_token=hf_token
)

# Disable NSFW checker (optional)
pipe.safety_checker = None

print("✅ LoRA model loaded and ready!")

@app.route("/gen", methods=["GET"])
def generate_image():
    prompt = request.args.get("prompt", "portrait of Shohan in cinematic light")
    try:
        result = pipe(prompt, num_inference_steps=30, guidance_scale=7.5)
        image = result.images[0]

        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({
            "prompt": prompt,
            "image": f"data:image/png;base64,{img_base64}"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
