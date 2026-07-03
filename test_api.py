import os
import sys

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

filepath = "Reference/transcripts/第7回_transcript.txt"
keywords = ["パッチクランプ", "どこでもドア", "スモールワールド", "スケール", "繋", "ランダム"]

with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

print(f"=== 第7回 transcript length: {len(content)} ===")
for keyword in keywords:
    idx = 0
    print(f"\n==================== Keyword: '{keyword}' ====================")
    count = 0
    while True:
        idx = content.find(keyword, idx)
        if idx == -1:
            break
        start = max(0, idx - 150)
        end = min(len(content), idx + 250)
        snippet = content[start:end]
        snippet_clean = snippet.encode('utf-8', errors='ignore').decode('utf-8')
        print(f"\n[{count}] (pos: {idx})")
        print(f"...{snippet_clean}...")
        idx += len(keyword) + 150
        count += 1
        if count >= 8: # あまりに多い場合は制限
            break
