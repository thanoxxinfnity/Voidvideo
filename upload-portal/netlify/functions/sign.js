// Netlify Function: returns a signed-upload payload for the browser to POST
// directly to Cloudinary. The API secret never leaves the server.
const crypto = require("crypto");

const ALLOWED_TAGS = ["locomotion", "gestures", "expressions", "secondary-motion"];

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

  const { tag } = body;
  if (!ALLOWED_TAGS.includes(tag)) {
    return json(400, { error: `tag must be one of: ${ALLOWED_TAGS.join(", ")}` });
  }

  const timestamp = Math.round(Date.now() / 1000);
  const folder = `voidvideo/${tag}`;

  const paramsToSign = { folder, tags: tag, timestamp };
  const toSign = Object.keys(paramsToSign)
    .sort()
    .map((k) => `${k}=${paramsToSign[k]}`)
    .join("&");
  const signature = crypto.createHash("sha1").update(toSign + apiSecret).digest("hex");

  return json(200, { cloudName, apiKey, timestamp, signature, folder, tags: tag });
};

function json(statusCode, obj) {
  return { statusCode, headers: { "Content-Type": "application/json" }, body: JSON.stringify(obj) };
}
