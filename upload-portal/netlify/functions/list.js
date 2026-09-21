// Netlify Function: lists uploaded video clips for a given category tag,
// using the Cloudinary Admin API (Basic Auth, server-side only).
const ALLOWED_TAGS = ["locomotion", "gestures", "expressions", "secondary-motion"];

exports.handler = async (event) => {
  if (event.httpMethod !== "GET") {
    return json(405, { error: "Method not allowed" });
  }

  const cloudName = process.env.CLOUDINARY_CLOUD_NAME;
  const apiKey = process.env.CLOUDINARY_API_KEY;
  const apiSecret = process.env.CLOUDINARY_API_SECRET;
  if (!cloudName || !apiKey || !apiSecret) {
    return json(500, { error: "Cloudinary environment variables are not configured on the server." });
  }

  const tag = event.queryStringParameters?.tag;
  if (!ALLOWED_TAGS.includes(tag)) {
    return json(400, { error: `tag must be one of: ${ALLOWED_TAGS.join(", ")}` });
  }

  const auth = Buffer.from(`${apiKey}:${apiSecret}`).toString("base64");
  const url = `https://api.cloudinary.com/v1_1/${cloudName}/resources/video/tags/${encodeURIComponent(tag)}?max_results=200`;

  try {
    const cloudinaryRes = await fetch(url, { headers: { Authorization: `Basic ${auth}` } });
    const data = await cloudinaryRes.json();
    if (!cloudinaryRes.ok) {
      return json(cloudinaryRes.status, { error: data.error?.message || "Cloudinary error" });
    }

    const clips = (data.resources || []).map((r) => ({
      publicId: r.public_id,
      url: r.secure_url,
      thumbnailUrl: r.secure_url.replace(/\.[a-zA-Z0-9]+$/, ".jpg").replace("/upload/", "/upload/so_0/"),
      bytes: r.bytes,
      duration: r.duration,
      createdAt: r.created_at,
    }));
    return json(200, { clips });
  } catch (err) {
    return json(502, { error: `Failed to reach Cloudinary: ${err.message}` });
  }
};

function json(statusCode, obj) {
  return { statusCode, headers: { "Content-Type": "application/json" }, body: JSON.stringify(obj) };
}
