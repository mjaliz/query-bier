#!/usr/bin/env python3
"""Test streaming threshold tuning"""

import asyncio
import json
import aiohttp

async def test_streaming():
    url = "http://localhost:5003/api/threshold-tuning"
    
    payload = {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "thresholds": [0.3, 0.5, 0.7],
        "batch_size": 32,
        "use_filtered_corpus": True,
        "max_queries": 10
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            print(f"Response status: {response.status}")
            
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        if data.get('type') == 'log':
                            print(f"[LOG] {data.get('message')}")
                        elif data.get('type') == 'progress':
                            print(f"[PROGRESS {data.get('progress')}%] {data.get('message')}")
                        elif data.get('type') == 'complete':
                            print("[COMPLETE] Evaluation finished!")
                            print(f"Best threshold: {data['results']['best_threshold']}")
                        elif data.get('type') == 'error':
                            print(f"[ERROR] {data.get('message')}")
                    except json.JSONDecodeError:
                        pass

if __name__ == "__main__":
    asyncio.run(test_streaming())