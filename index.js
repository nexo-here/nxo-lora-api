require("dotenv").config();
const express = require("express");
const cors = require("cors");
const { DiffusionPipeline } = require("@xenova/transformers");

const app = express();
app.use(cors());

const port = process.env.PORT || 3000;

let pipe = null;

(async () => {
  console.log("🔄 Loading base model...");
  pipe = await DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", {
    use_auth_token: process.env.HUGGINGFACE_TOKEN,
  });

  console.log("🔄 Applying LoRA...");
  await pipe.load_lora_weights("nexo-here/shohan-lora", {
    weight_name: "shohan-lora.safetensors",
    use_auth_token: process.env.HUGGINGFACE_TOKEN,
  });

  console.log("✅ LoRA applied, ready to generate!");
})();

app.get("/gen", async (req, res) => {
  const prompt = req.query.prompt || "Shohan in anime style";

  try {
    if (!pipe) {
      return res.status(503).json({ error: "Model is still loading" });
    }

    const output = await pipe(prompt, {
      guidance_scale: 7,
      num_inference_steps: 25,
    });

    const image = output.images[0];
    const base64 = image.toString("base64");

    res.json({
      image: `data:image/png;base64,${base64}`,
    });

  } catch (e) {
    console.error("❌ Image generation error:", e);
    res.status(500).json({ error: "Image generation failed" });
  }
});

app.listen(port, () => {
  console.log(`🚀 Server running at http://localhost:${port}`);
});
