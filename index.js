const express = require("express");
const axios = require("axios");
const cors = require("cors");
const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.get("/", (req, res) => res.send("LoRA Image Gen API is live"));

app.get("/gen", async (req, res) => {
  const prompt = req.query.prompt || "Shohan in a beautiful garden";
  const loraName = req.query.lora || "Shohan";
  const modelUrl = req.query.model_url || "https://huggingface.co/nexo-here/shohan-lora/resolve/main/shohan-lora.safetensors";

  try {
    const genRes = await axios.get(`https://api.segmind.com/v1/sdxl-lora`, {
      headers: {
        "x-api-key": "SG_8fe2efd5839e7b47"
      },
      params: {
        prompt: `${loraName}, ${prompt}`,
        lora_urls: modelUrl,
        num_inference_steps: 28,
        guidance_scale: 7
      }
    });

    return res.json({ image: genRes.data.image_url });
  } catch (e) {
    console.error(e.response?.data || e.message);
    return res.status(500).json({ error: "Generation failed" });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
