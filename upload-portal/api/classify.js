// Vercel serverless function: classifies a single video frame into one of the
// 4 clip categories using an NVIDIA NIM vision-language model. The API key
// never reaches the browser -- only a small JPEG frame is sent up here, and
// only a category slug goes back down.
const ALLOWED_TAGS = ["locomotion", "gestures", "expressions", "secondary-motion"];

const PROMPT = `You are sorting short reference video clips for an animation dataset into exactly one of these 4 categories:

- locomotion: walking, running, jogging, idle standing/breathing, turning around, changing stance
- gestures: waving, pointing, reaching for something, picking up an object, opening a door, using a phone, hand/arm actions
- expressions: close-up facial expressions -- blinking, smiling, frowning, surprise, talking, anger
- secondary-motion: passive ambient motion with no deliberate human action -- hair swaying, cloth/fabric moving, leaves falling, water rippling, a curtain moving

Look at this single frame from the middle of the clip and pick the single best-matching category.
Reply with ONLY one lowercase word, exactly one of: locomotion, gestures, expressions, secondary-motion, unsure
No punctuation, no explanation, no other text.`;

module.exports = async (req, res) => {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const apiKey = process.env.NVIDIA_API_KEY;
  if (!apiKey) {
    res.status(500).json({ error: "NVIDIA_API_KEY is not configured on the server." });
    return;
  }

  const { imageBase64 } = req.body || {};
  if (!imageBase64 || typeof imageBase64 !== "string") {
    res.status(400).json({ error: "imageBase64 (base64 JPEG, no data: prefix) is required" });
    return;
  }
  // A single downscaled JPEG frame comfortably fits NVIDIA's inline base64
  // limit; anything bigger means the client didn't downscale as expected.
  if (imageBase64.length > 700_000) {
    res.status(400).json({ error: "Image too large -- downscale the frame before sending it." });
    return;
  }

  try {
    const upstream = await fetch("https://integrate.api.nvidia.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        model: "meta/llama-3.2-11b-vision-instruct",
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: PROMPT },
              { type: "image_url", image_url: { url: `data:image/jpeg;base64,${imageBase64}` } },
            ],
          },
        ],
        max_tokens: 16,
        temperature: 0.0,
      }),
    });

    const data = await upstream.json();
    if (!upstream.ok) {
      res.status(upstream.status).json({ error: data.error?.message || data.detail || "NVIDIA API error" });
      return;
    }

    const raw = (data.choices?.[0]?.message?.content || "").trim().toLowerCase();
    const match = ALLOWED_TAGS.find((tag) => raw.includes(tag));
    res.status(200).json({ category: match || null, raw });
  } catch (err) {
    res.status(502).json({ error: `Failed to reach NVIDIA API: ${err.message}` });
  }
};
