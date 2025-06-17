const express = require("express");
const axios = require("axios");
const cors = require("cors");
require("dotenv").config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());

app.get("/", (req, res) => res.send("✅ LoRA API is live. Use /gen?prompt=your_prompt"));

app.get("/gen", async (req, res) => {
  const prompt = req.query.prompt || "Shohan in a fantasy garden";
  const loraUrl = "https://huggingface.co/nexo-here/shohan-lora/resolve/main/shohan-lora.safetensors";

  try {
    const genRes = await axios.get("https://api.segmind.com/v1/sdxl-lora", {
      headers: {
        "x-api-key": process.env.SEGMIND_API_KEY
      },
      params: {
        prompt: `Shohan, ${prompt}`,
        lora_urls: loraUrl,
        num_inference_steps: 28,
        guidance_scale: 7
      }
    });

    res.json({ image: genRes.data.image_url });
  } catch (err) {
    console.error(err.response?.data || err.message);
    res.status(500).json({ error: "Image generation failed" });
  }
});

app.listen(PORT, () => {
  console.log(`✅ Server running on http://localhost:${PORT}`);
});
