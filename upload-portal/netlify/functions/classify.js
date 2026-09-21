// Netlify Function: classifies a single video frame into one of the 4 clip
// categories using an NVIDIA NIM vision-language model, server-side only.
const ALLOWED_TAGS = ["locomotion", "gestures", "expressions", "secondary-motion"];

const PROMPT = `You are sorting short reference video clips for an animation dataset into exactly one of these 4 categories:

- locomotion: walking, running, jogging, idle standing/breathing, turning around, changing stance
- gestures: waving, pointing, reaching for something, picking up an object, opening a door, using a phone, hand/arm actions
- expressions: close-up facial expressions -- blinking, smiling, frowning, surprise, talking, anger
- secondary-motion: passive ambient motion with no deliberate human action -- hair swaying, cloth/fabric moving, leaves falling, water rippling, a curtain moving

Look at this single frame from the middle of the clip and pick the single best-matching category.
Reply with ONLY one lowercase word, exactly one of: locomotion, gestures, expressions, secondary-motion, unsure
No punctuation, no explanation, no other text.`;

exports.handler = async (event) => {
  if (event.httpMethod !== "POST") {
    return json(405, { error: "Method not allowed" });
  }

  const apiKey = process.env.NVIDIA_API_KEY;
  if (!apiKey) {
    return json(500, { error: "NVIDIA_API_KEY is not configured on the server." });
  }

  let body;
  try {
    body = JSON.parse(event.body || "{}");
  } catch {
    return json(400, { error: "Invalid JSON body" });
  }

  const { imageBase64 } = body;
  if (!imageBase64 || typeof imageBase64 !== "string") {
    return json(400, { error: "imageBase64 (base64 JPEG, no data: prefix) is required" });
  }
  if (imageBase64.length > 700_000) {
    return json(400, { error: "Image too large -- downscale the frame before sending it." });
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
      return json(upstream.status, { error: data.error?.message || data.detail || "NVIDIA API error" });
    }

    const raw = (data.choices?.[0]?.message?.content || "").trim().toLowerCase();
    const match = ALLOWED_TAGS.find((tag) => raw.includes(tag));
    return json(200, { category: match || null, raw });
  } catch (err) {
    return json(502, { error: `Failed to reach NVIDIA API: ${err.message}` });
  }
};

function json(statusCode, obj) {
  return { statusCode, headers: { "Content-Type": "application/json" }, body: JSON.stringify(obj) };
}
