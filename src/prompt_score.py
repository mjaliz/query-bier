instruction = """

You are an expert in evaluating search relevance for Persian e--commerce. Your task is to analyze the relationship between a single search `query` and a list of `corpus` items. For each item in the list, you must assign a relevance score based on the detailed guide below and return the results as a single JSON list.

**Inputs:**
*   **Query:** `{query}`
*   **Corpus List:** `{corpus_list}`

**Task:**
For each object in the input `corpus_list`, compare its `corpus_text` to the provided `query` and assign a relevance score. Your output must be a valid JSON list of objects, where each object contains the `corpus_id` and `corpus_text` from the input, plus the `score` you assigned.

**Input Corpus List Format:**
```json
[
    {{
        "corpus_id": "c_3624",
        "corpus_text": "پارچه ملحفه ای"
    }},
    {{
        "corpus_id": "c_89",
        "corpus_text": "انواع پارچه ملافه"
    }}
]
```

**Required Output Format:**
The output must be a JSON list where each element corresponds to an item from the input list.
```json
[
  {{
    "corpus_id": "c_3624",
    "corpus_text": "پارچه ملحفه ای",
    "score": <score_integer_1>
  }},
  {{
    "corpus_id": "c_89",
    "corpus_text": "انواع پارچه ملافه",
    "score": <score_integer_2>
  }}
]
```
Replace `<score_integer_...>` with the calculated integer score (1, 2, 3, or 4) for each item.

**Relevance Scoring Guide (Higher score means more relevant):**

*   **1: Lexically Similar, Semantically Different:** The items share significant lexical overlap (words) but refer to entirely different products or concepts.
    *   *Example for "نخ دندان" (dental floss) vs. "نخ خیاطی" (sewing thread).*
*   **2: Keyword Matching & Broader/Narrower Categories:** The items share one or more key terms, and the `corpus_text` represents a related, but not directly synonymous, category (broader, narrower, or sibling).
    *   *Example for "برنج هاشمی" (Hashemi rice) vs. "برنج ایرانی" (Iranian rice).*
*   **3: Semantically Similar, Lexically Different:** The `corpus_text` is functionally or conceptually very close to the `query` (a strong synonym or alternative name) but uses entirely different vocabulary.
    *   *Example for "جاروبرقی صورت" (face vacuum cleaner) vs. "میکرودرم" (microdermabrasion device).*
*   **4: Highly Relevant / Exact Match / Close Synonym:** The `corpus_text` is the query itself, an exact synonym, or a very close variation that is functionally identical.
    *   *Example for "نخ دندان" (dental floss) vs. "نخ دندان فلورایددار" (fluoridated dental floss).*

**--- IMPORTANT CONTEXT RULE ---**
Your evaluation should focus purely on the product names, concepts, or categories. You must ignore any transactional or commercial intent keywords (e.g., `خرید`, `قیمت`). For example, evaluate "خرید گوشی سامسونگ" against "قیمت گوشی سامسونگ A54" by scoring the relationship between "گوشی سامسونگ" and "گوشی سامسونگ A54".
**--- END OF RULE ---**

Generate the JSON output immediately, following all instructions precisely.

"""
