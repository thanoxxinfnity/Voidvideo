// Vercel serverless function: deletes an uploaded clip by public_id via the
// Cloudinary Admin API (Basic Auth, server-side only).
module.exports = async (req, res) => {
  if (req.method !== "POST") {
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

  const { publicId } = req.body || {};
  // public_ids from our own /api/list response always start with "voidvideo/"
  // (our upload folder prefix) -- reject anything else so this endpoint can't
  // be used to delete arbitrary resources elsewhere in the Cloudinary account.
  if (typeof publicId !== "string" || !publicId.startsWith("voidvideo/")) {
    res.status(400).json({ error: "Invalid publicId" });
    return;
  }

  const auth = Buffer.from(`${apiKey}:${apiSecret}`).toString("base64");
  const url = `https://api.cloudinary.com/v1_1/${cloudName}/resources/video/upload?public_ids[]=${encodeURIComponent(publicId)}`;

  try {
    const cloudinaryRes = await fetch(url, {
      method: "DELETE",
      headers: { Authorization: `Basic ${auth}` },
    });
    const data = await cloudinaryRes.json();
    if (!cloudinaryRes.ok) {
      res.status(cloudinaryRes.status).json({ error: data.error?.message || "Cloudinary error" });
      return;
    }
    res.status(200).json({ deleted: data.deleted || {} });
  } catch (err) {
    res.status(502).json({ error: `Failed to reach Cloudinary: ${err.message}` });
  }
};
