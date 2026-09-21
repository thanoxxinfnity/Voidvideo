// Vercel serverless function: lists uploaded video clips for a given category
// tag, using the Cloudinary Admin API (Basic Auth with key:secret, server-side only).
const ALLOWED_TAGS = ["locomotion", "gestures", "expressions", "secondary-motion"];

module.exports = async (req, res) => {
  if (req.method !== "GET") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const cloudName = process.env.CLOUDINARY_CLOUD_NAME;
  const apiKey = process.env.CLOUDINARY_API_KEY;
  const apiSecret = process.env.CLOUDINARY_API_SECRET;
  if (!cloudName || !apiKey || !apiSecret) {
    res.status(500).json({ error: "Cloudinary environment variables are not configured on the server." });
    return;
  }

  const tag = req.query.tag;
  if (!ALLOWED_TAGS.includes(tag)) {
    res.status(400).json({ error: `tag must be one of: ${ALLOWED_TAGS.join(", ")}` });
    return;
  }

  const auth = Buffer.from(`${apiKey}:${apiSecret}`).toString("base64");
  const url = `https://api.cloudinary.com/v1_1/${cloudName}/resources/video/tags/${encodeURIComponent(tag)}?max_results=200&context=true`;

  try {
    const cloudinaryRes = await fetch(url, { headers: { Authorization: `Basic ${auth}` } });
    const data = await cloudinaryRes.json();
    if (!cloudinaryRes.ok) {
      res.status(cloudinaryRes.status).json({ error: data.error?.message || "Cloudinary error" });
      return;
    }

    const clips = (data.resources || []).map((r) => ({
      publicId: r.public_id,
      url: r.secure_url,
      thumbnailUrl: r.secure_url.replace(/\.[a-zA-Z0-9]+$/, ".jpg").replace("/upload/", "/upload/so_0/"),
      bytes: r.bytes,
      duration: r.duration,
      createdAt: r.created_at,
      caption: r.context?.custom?.caption || null,
    }));
    res.status(200).json({ clips });
  } catch (err) {
    res.status(502).json({ error: `Failed to reach Cloudinary: ${err.message}` });
  }
};
