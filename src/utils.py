import json
from pathlib import Path

import pandas as pd


def gen_id():
    json_files = Path(__file__).parent.parent.glob("*.json")
    data = []
    for file in json_files:
        with open(file, "r") as f:
            data.extend(json.loads(f.read()))
    k = 1
    j = 1
    for i in range(0, len(data)):
        data[i]["_id"] = f"q_{k}"
        k += 1
        for q in range(0, len(data[i]["curpos"])):
            data[i]["curpos"][q]["_id"] = f"c_{j}"
            j += 1
    with open("identified_data.json", "w") as f:
        f.write(json.dumps(data, ensure_ascii=False))


def build_bire():
    path = Path(__file__).parent.parent / "identified_data.json"
    with open(path, "r") as f:
        data = json.loads(f.read())

    queries = []
    corpus = []
    q_id = []
    c_id = []
    score = []
    for item in data:
        queries.append({"_id": item["_id"], "text": item["query"]})
        for c in item["curpos"]:
            q_id.append(item["_id"])
            c_id.append(c["_id"])
            s = 1 if c["score"] >= 3 else 0
            score.append(s)
            corpus.append({"_id": c["_id"], "text": c["text"], "title": c["text"]})
    with open("corpus.jsonl", "w") as f:
        f.write("\n".join(map(lambda x: json.dumps(x, ensure_ascii=False), corpus)))

    with open("query.jsonl", "w") as f:
        f.write("\n".join(map(lambda x: json.dumps(x, ensure_ascii=False), queries)))

    df = pd.DataFrame({"q": q_id, "doc": c_id, "score": score})
    df.to_csv("qrels.tsv", sep="\t", index=False)


if __name__ == "__main__":
    gen_id()
    build_bire()
