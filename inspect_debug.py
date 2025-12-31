
import json
import os
import glob

# Find latest debug dir
debug_dirs = glob.glob("debug/*")
debug_dirs.sort(key=os.path.getmtime, reverse=True)

if not debug_dirs:
    print("No debug directories found.")
    exit()

latest_dir = debug_dirs[0]
info_path = os.path.join(latest_dir, "info.json")

print(f"Reading {info_path}")

try:
    with open(info_path, 'r') as f:
        data = json.load(f)
        
    with open("debug_summary.txt", "w") as out:
        out.write(f"Match: {data.get('match')}\n")
        out.write("Candidates:\n")
        candidates = data.get("candidates", [])
        for i, c in enumerate(candidates):
            out.write(f"  {i+1}. {c['name']} (Score: {c.get('score'):.4f}, Global: {c.get('global_score'):.4f})\n")
            out.write(f"     Learned: {c.get('learned', False)}\n")
            out.write(f"     Spatial: {c.get('spatial_verification', 0)}\n")
            out.write(f"     Image: {c.get('image')}\n")
            out.write("\n")

    print("Summary written to debug_summary.txt")

except Exception as e:
    print(f"Error: {e}")
