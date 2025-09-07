instruction = """"
You are an expert in evaluating search relevance for Persian e-commerce. Your task is to analyze the relationship between a given search `query` and a `corpus_item`, assign a single relevance score based on the detailed guide below, and return the result in a specific JSON format.

**Inputs:**
*   **Query:** `{query}`
*   **Corpus Item:** `{corpus_item}`

**Task:**
Based on the scoring guide, determine the relevance score between the provided `query` and `corpus_item`. Your output must be a valid JSON object containing the original query, the original corpus item, and the assigned score.

The JSON output must strictly adhere to this format:
```json
{{
  "query": "{query}",
  "corpus": "{corpus_item}",
  "score": <score_integer>
}}
```
Replace `<score_integer>` with the calculated integer score: 1, 2, 3, or 4.

**Relevance Scoring Guide (Higher score means more relevant):**

*   **1: Lexically Similar, Semantically Different:** The items share significant lexical overlap (words) but refer to entirely different products or concepts.
    *   *Example for "نخ دندان" (dental floss) vs. "نخ خیاطی" (sewing thread).*
    *   *Example for "چادر ماشین" (car cover) vs. "چادر مشکی" (black tent/chador).*
*   **2: Keyword Matching & Broader/Narrower Categories:** The items share one or more key terms, and the `corpus_item` represents a related, but not directly synonymous, category. This includes broader categories, narrower specifications, or sibling items.
    *   *Example for "برنج هاشمی" (Hashemi rice) vs. "برنج ایرانی" (Iranian rice).*
*   **3: Semantically Similar, Lexically Different:** The `corpus_item` is functionally or conceptually very close to the `query` (a strong synonym or alternative name) but uses entirely different vocabulary.
    *   *Example for "جاروبرقی صورت" (face vacuum cleaner) vs. "میکرودرم" (microdermabrasion device).*
    *   *Example for "اسکوچی" (Scoochie/electric scooter) vs. "اسکوتر برقی" (electric scooter).*
*   **4: Highly Relevant / Exact Match / Close Synonym:** The `corpus_item` is the query itself, an exact synonym, or a very close variation that is functionally identical (e.g., includes a minor specifier like flavor, color, or a standard feature).
    *   *Example for "نخ دندان" (dental floss) vs. "نخ دندان فلورایددار" (fluoridated dental floss).*

**--- IMPORTANT CONTEXT RULE ---**
Your evaluation should focus purely on the product names, concepts, or categories themselves. You must ignore any transactional or commercial intent keywords that might appear in the strings. For example, if evaluating "خرید گوشی سامسونگ" (buy Samsung phone) against "قیمت گوشی سامسونگ A54" (price of Samsung A54 phone), you should effectively score the relationship between "گوشی سامسونگ" and "گوشی سامسونگ A54". Disregard modifiers like:
*   `خرید` (buy)
*   `قیمت` (price)
*   `فروش` (sale)
*   `ارزان` (cheap)
*   `مشخصات` (specifications)
**--- END OF RULE ---**

Generate the JSON output immediately, following all instructions and formatting guidelines precisely.
"""
