// Vercel serverless function: writes a short training caption for a clip
// using a vision-language model via Hugging Face's hosted Inference
// Providers. HF_TOKEN never reaches the browser -- only the same downscaled
// contact-sheet JPEG already sent to /api/classify goes up here.
const HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions";
// Qwen2-VL-7B-Instruct has no hosted inference provider at all, and
// Qwen2.5-VL-7B-Instruct's only provider (featherless-ai) was frequently at
// capacity when this was built. Qwen2.5-VL-72B-Instruct via ovhcloud was
// verified working (including genuine multi-frame motion reasoning) and is
// the default -- override with HF_CAPTION_MODEL if your account has a
// different provider you'd rather use.
const HF_CAPTION_MODEL = process.env.HF_CAPTION_MODEL || "Qwen/Qwen2.5-VL-72B-Instruct:ovhcloud";

const PROMPT = `You are writing a short training caption for an anime video LoRA dataset. This image is a contact sheet of frames sampled evenly across ONE short video clip, in time order -- read them as a timeline, not separate pictures. In ONE concise sentence, describe: what the character is doing (e.g. walking, waving, blinking, hair/cloth moving in the wind), their facial expression if visible, and the camera shot (close-up/medium/wide, static or panning). If no character is visible, describe what is moving instead. Do not mention "frame" or "image" -- describe it as a continuous video clip. Reply with ONLY the caption sentence, no preamble.`;

module.exports = async (req, res) => {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const apiKey = process.env.HF_TOKEN;
  if (!apiKey) {
    res.status(500).json({ error: "HF_TOKEN is not configured on the server." });
    return;
  }

  const { imageBase64 } = req.body || {};
  if (!imageBase64 || typeof imageBase64 !== "string") {
    res.status(400).json({ error: "imageBase64 (base64 JPEG, no data: prefix) is required" });
    return;
  }
  if (imageBase64.length > 700_000) {
    res.status(400).json({ error: "Image too large -- downscale the frame before sending it." });
    return;
  }

  try {
    const upstream = await fetch(HF_ROUTER_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: HF_CAPTION_MODEL,
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: PROMPT },
              { type: "image_url", image_url: { url: `data:image/jpeg;base64,${imageBase64}` } },
            ],
          },
        ],
        max_tokens: 120,
        temperature: 0.2,
      }),
    });

    const data = await upstream.json();
    if (!upstream.ok) {
      const msg = (data.error || {}).message || data.message || "Hugging Face API error";
      res.status(upstream.status).json({ error: msg });
      return;
    }

    const caption = (data.choices?.[0]?.message?.content || "").trim();
    res.status(200).json({ caption: caption || null });
  } catch (err) {
    res.status(502).json({ error: `Failed to reach Hugging Face: ${err.message}` });
  }
};
