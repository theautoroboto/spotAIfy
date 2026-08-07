import subprocess, os, sys
from dotenv import dotenv_values

# dotenv_values preserves $$; Docker Compose unescapes $$ → $ before injecting
# into the container, so replicate that here.
raw = dotenv_values(".env")
parsed = {k: v.replace("$$", "$") if v else v for k, v in raw.items()}
env = {**os.environ, **parsed}
env["DB_PATH"] = r"D:\git\spotAIfy\spotaify-dev.db"
env["BG_CACHE_DIR"] = r"D:\git\spotAIfy\data\backgrounds"
env["AVATAR_CACHE_DIR"] = r"D:\git\spotAIfy\data\avatars"
env["HISTORY_DIR"] = r"D:\git\spotAIfy\data\history"

proc = subprocess.Popen(
    ["go", "run", "./cmd/web"],
    env=env,
    cwd=r"D:\git\spotAIfy",
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
for line in proc.stdout:
    print(line, end="", flush=True)
