// Netlify Function: writes a short training caption for a clip using a
// vision-language model via Hugging Face's hosted Inference Providers,
// server-side only.
const HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions";
const HF_CAPTION_MODEL = process.env.HF_CAPTION_MODEL || "Qwen/Qwen2.5-VL-72B-Instruct:ovhcloud";

const PROMPT = `You are writing a short training caption for an anime video LoRA dataset. This image is a contact sheet of frames sampled evenly across ONE short video clip, in time order -- read them as a timeline, not separate pictures. In ONE concise sentence, describe: what the character is doing (e.g. walking, waving, blinking, hair/cloth moving in the wind), their facial expression if visible, and the camera shot (close-up/medium/wide, static or panning). If no character is visible, describe what is moving instead. Do not mention "frame" or "image" -- describe it as a continuous video clip. Reply with ONLY the caption sentence, no preamble.`;

exports.handler = async (event) => {
  if (event.httpMethod !== "POST") {
    return json(405, { error: "Method not allowed" });
  }

  const apiKey = process.env.HF_TOKEN;
  if (!apiKey) {
    return json(500, { error: "HF_TOKEN is not configured on the server." });
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
      return json(upstream.status, { error: msg });
    }

    const caption = (data.choices?.[0]?.message?.content || "").trim();
    return json(200, { caption: caption || null });
  } catch (err) {
    return json(502, { error: `Failed to reach Hugging Face: ${err.message}` });
  }
};

function json(statusCode, obj) {
  return { statusCode, headers: { "Content-Type": "application/json" }, body: JSON.stringify(obj) };
}
