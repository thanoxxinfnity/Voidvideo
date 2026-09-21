// Vercel serverless function: classifies a video into one of the 4 clip
// categories using an NVIDIA NIM vision-language model. The API key never
// reaches the browser -- only a small JPEG goes up here, and only a category
// slug comes back down.
//
// The model available to this account (meta/llama-3.2-11b-vision-instruct)
// hard-rejects more than one image per request, so a single static frame
// can't carry any motion information -- and motion is exactly what tells
// these categories apart. The client works around this by tiling several
// frames sampled across the whole clip into one contact-sheet image (see
// extractFrameGridBase64 in app.js), so this is judging the clip's motion
// over time, not one instant of it.
const ALLOWED_TAGS = ["locomotion", "gestures", "expressions", "secondary-motion"];

// Forcing an immediate single-word answer (earlier version) made this model
// default to "locomotion" almost every time, including for a clip of falling
// leaves with no character in it at all. Letting it describe the scene in
// one sentence FIRST, then classify based on its own description, fixed
// this in testing -- small vision models follow a decision procedure much
// more reliably when they "think out loud" before answering.
const PROMPT = `These images are frames sampled evenly across a single short anime video clip, in time order left-to-right then top-to-bottom (read it as a timeline).

First, in one short sentence, describe exactly what is moving in this clip and whether a character's body/hand/face is the main subject, or whether nothing in frame is a character at all.

Then, on a new line, write "CATEGORY: " followed by exactly one of these 4 words based on your own description above:
- secondary-motion -- ONLY if no character is visible in the clip at all (e.g. just falling leaves, rippling water, an empty room)
- gestures -- if a character's hand/arm deliberately reaches, grabs, points, waves, or opens something
- expressions -- if a character IS visible and their face shows a clear emotion (smile, frown, surprise, blink, talk) -- pick this even if their hair is also blowing in the wind, the hair is not the point
- locomotion -- if a character IS visible and their whole body is walking, running, turning, or idly standing with no particular expression or hand action being the point

Example output:
A red leaf is falling through the air with no person visible.
CATEGORY: secondary-motion`;

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
        max_tokens: 80,
        temperature: 0.0,
      }),
    });

    const data = await upstream.json();
    if (!upstream.ok) {
      res.status(upstream.status).json({ error: data.error?.message || data.detail || "NVIDIA API error" });
      return;
    }

    const raw = (data.choices?.[0]?.message?.content || "").trim().toLowerCase();
    // The reply is now a description sentence + "CATEGORY: <word>" -- search
    // after the last "category:" marker so a coincidental word earlier in
    // the description can't be mistaken for the actual answer.
    const markerIdx = raw.lastIndexOf("category:");
    const searchText = markerIdx >= 0 ? raw.slice(markerIdx) : raw;
    const match = ALLOWED_TAGS.find((tag) => searchText.includes(tag));
    res.status(200).json({ category: match || null, raw });
  } catch (err) {
    res.status(502).json({ error: `Failed to reach NVIDIA API: ${err.message}` });
  }
};
