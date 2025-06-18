import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { pipeline } from '@xenova/transformers';

const app = express();
app.use(cors());
const port = process.env.PORT || 3000;

let pipe = null;

(async () => {
  console.log("🔄 Loading base model...");
  pipe = await pipeline("text-to-image", "Xenova/stable-diffusion-v1-5", {
    use_auth_token: process.env.HUGGINGFACE_TOKEN
  });

  console.log("🔄 Applying LoRA...");
  await pipe.load_lora_weights("nexo-here/shohan-lora", {
    weight_name: "shohan-lora.safetensors",
    use_auth_token: process.env.HUGGINGFACE_TOKEN
  });

  console.log("✅ Ready for generation!");
})();

app.get("/gen", async (req, res) => {
  const prompt = req.query.prompt || "Shohan in anime style";
  try {
    if (!pipe) return res.status(503).json({ error: "Model loading..." });

    const result = await pipe(prompt, {
      guidance_scale: 7.5,
      num_inference_steps: 25,
    });

    const image = result.images[0];
    const base64 = image.toString("base64");

    res.json({ image: `data:image/png;base64,${base64}` });
  } catch (err) {
    console.error("❌ Generation failed:", err);
    res.status(500).json({ error: "Image generation failed" });
  }
});

app.listen(port, () => {
  console.log(`🚀 Server running at http://localhost:${port}`);
});
