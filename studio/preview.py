"""Small JPEG preview of a finished clip (middle frame, <= 320 px wide) so the phone shows a thumbnail without downloading the mp4."""
import subprocess


def mid_frame(mp4, dst, width=320):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-vf", f"select='eq(n\\,12)',scale={width}:-2", "-frames:v", "1",
                    "-q:v", "5", str(dst)], check=True)
    return str(dst)
