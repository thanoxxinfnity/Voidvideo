// Netlify Function: deletes an uploaded clip by public_id via the Cloudinary
// Admin API (Basic Auth, server-side only).
exports.handler = async (event) => {
  if (event.httpMethod !== "POST") {
    return json(405, { error: "Method not allowed" });
  }

  const cloudName = process.env.CLOUDINARY_CLOUD_NAME;
  const apiKey = process.env.CLOUDINARY_API_KEY;
  const apiSecret = process.env.CLOUDINARY_API_SECRET;
  if (!cloudName || !apiKey || !apiSecret) {
    return json(500, { error: "Cloudinary environment variables are not configured on the server." });
  }

  let body;
  try {
    body = JSON.parse(event.body || "{}");
  } catch {
    return json(400, { error: "Invalid JSON body" });
  }

  const { publicId } = body;
  // public_ids from our own list function always start with "voidvideo/" (our
  // upload folder prefix) -- reject anything else so this can't delete
  // arbitrary resources elsewhere in the Cloudinary account.
  if (typeof publicId !== "string" || !publicId.startsWith("voidvideo/")) {
    return json(400, { error: "Invalid publicId" });
  }

  const auth = Buffer.from(`${apiKey}:${apiSecret}`).toString("base64");
  const url = `https://api.cloudinary.com/v1_1/${cloudName}/resources/video/upload?public_ids[]=${encodeURIComponent(publicId)}`;

  try {
    const cloudinaryRes = await fetch(url, { method: "DELETE", headers: { Authorization: `Basic ${auth}` } });
    const data = await cloudinaryRes.json();
    if (!cloudinaryRes.ok) {
      return json(cloudinaryRes.status, { error: data.error?.message || "Cloudinary error" });
    }
    return json(200, { deleted: data.deleted || {} });
  } catch (err) {
    return json(502, { error: `Failed to reach Cloudinary: ${err.message}` });
  }
};

function json(statusCode, obj) {
  return { statusCode, headers: { "Content-Type": "application/json" }, body: JSON.stringify(obj) };
}
