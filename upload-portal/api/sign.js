// Vercel serverless function: returns a signed-upload payload for the browser
// to POST directly to Cloudinary. The API secret never leaves the server.
const crypto = require("crypto");

const ALLOWED_TAGS = ["locomotion", "gestures", "expressions", "secondary-motion"];

module.exports = async (req, res) => {
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  const { cloudName, apiKey, apiSecret } = readCloudinaryEnv();
  if (!cloudName || !apiKey || !apiSecret) {
    res.status(500).json({ error: "Cloudinary environment variables are not configured on the server." });
    return;
  }

  const { tag } = req.body || {};
  if (!ALLOWED_TAGS.includes(tag)) {
    res.status(400).json({ error: `tag must be one of: ${ALLOWED_TAGS.join(", ")}` });
    return;
  }

  const timestamp = Math.round(Date.now() / 1000);
  const folder = `voidvideo/${tag}`;

  // Cloudinary signature: sha1 of alphabetically-sorted "key=value" pairs
  // (excluding file/api_key/signature/resource_type) + api_secret, no delimiters.
  const paramsToSign = { folder, tags: tag, timestamp };
  const toSign = Object.keys(paramsToSign)
    .sort()
    .map((k) => `${k}=${paramsToSign[k]}`)
    .join("&");
  const signature = crypto.createHash("sha1").update(toSign + apiSecret).digest("hex");

  res.status(200).json({
    cloudName,
    apiKey,
    timestamp,
    signature,
    folder,
    tags: tag,
  });
};

function readCloudinaryEnv() {
  return {
    cloudName: process.env.CLOUDINARY_CLOUD_NAME,
    apiKey: process.env.CLOUDINARY_API_KEY,
    apiSecret: process.env.CLOUDINARY_API_SECRET,
  };
}
