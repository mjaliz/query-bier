instruction = """
"You are an expert in building custom benchmarks for Persian e-commerce search. Your task is to generate a comprehensive and balanced list of related but distinct Persian corpus items (`curpos`) for a given search query, along with their relevance scores. The output must be a valid JSON list, where each element is a two-item list containing the corpus item (string) and its relevance score (integer).

The JSON output must strictly adhere to this format:
```json
{{
  "curpos": [
    {{"text": "corpus_item_1", "score": score_1}},
    {{"text": "corpus_item_2", "score": score_2}},
    {{"text": "corpus_item_3", "score": score_3}},
    // ... and so on
  ]
}}
```

**Relevance Scoring Guide (Higher score means more relevant):**

*   **1: Lexically Similar, Semantically Different:** Items that share significant lexical overlap with the query but refer to entirely different products or concepts.
    *   *Example for "نخ دندان" (dental floss):* "نخ خیاطی" (sewing thread).
    *   *Example for "چادر ماشین" (car cover):* "چادر مشکی" (black tent/chador).
*   **2: Keyword Matching & Broader/Narrower Categories:** Items that share one or more key terms with the query but represent a related, but not directly synonymous, category.
    *   *Example for "برنج هاشمی" (Hashemi rice):* "برنج ایرانی" (Iranian rice), "انواع برنج" (types of rice).
*   **3: Semantically Similar, Lexically Different:** Items that are functionally or conceptually very close to the query, but use entirely different vocabulary. These are often synonyms or alternative names.
    *   *Example for "جاروبرقی صورت" (face vacuum cleaner):* "میکرودرم" (microdermabrasion device).
    *   *Example for "اسکوچی" (Scoochie/electric scooter):* "اسکوتر برقی" (electric scooter).
*   **4: Highly Relevant / Exact Match / Close Synonym:** The query itself, exact synonyms, or very close variations that are functionally identical.
    *   *Example for "نخ دندان":* "نخ دندان فلورایددار" (fluoridated dental floss).

**Your Goal:**
For the given query, generate a balanced set of corpus items across all defined scoring categories to ensure the distribution does not affect the NGCD metric. You must generate **exactly 5 distinct corpus items for each of the 4 scoring categories** (1, 2, 3, and 4), resulting in a total of 20 items.

**--- IMPORTANT EXCLUSION RULE ---**
The generated `curpos` items **must be product names, concepts, or categories.** They should **NOT** contain transactional or commercial intent keywords. Specifically, avoid generating items that include words like:
*   `خرید` (buy)
*   `قیمت` (price)
*   `فروش` (sale)
*   `ارزان` (cheap)
*   `تخفیف` (discount)
*   `مشخصات` (specifications)
*   `لیست` (list)
*   ...or any similar commercial/informational modifiers. Focus purely on the items themselves.
**--- END OF RULE ---**

**Query:**
{query}

**Task:**
Generate the JSON output immediately, following all instructions and formatting guidelines precisely. Ensure the corpus items are natural Persian phrases or product names that adhere to the exclusion rule.
"
"""
