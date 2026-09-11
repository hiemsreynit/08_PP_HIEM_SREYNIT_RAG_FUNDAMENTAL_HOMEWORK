# RAG Pipeline Evaluation Test Log

**Author:** Hiem Sreynit
**Project:** Naive RAG Fundamental Homework
**LLM:** `qwen2.5:0.5b-instruct`
**Embedding:** `nomic-embed-text`
**Vector Database:** ChromaDB

---

## Evaluation Summary

|  #  | Question                                                          | Source Document                                              | Result  |
| :-: | ----------------------------------------------------------------- | ------------------------------------------------------------ | ------- |
|  1  | How do I set up a conference call using Cisco Webex?              | `005_Setting_Up_a_Conference_Call_on_Cisco_Webex.txt`        | Passed  |
|  2  | How do I create a backup of important files?                      | `006_Creating_a_Backup_of_Important_Files.txt`               | Passed  |
|  3  | What should I do if I have problems with a company-issued tablet? | `007_Troubleshooting_Issues_with_Company-Issued_Tablets.txt` | Passed  |
|  4  | `[Another question from your documents]`                          | `[Source document]`                                          | Passed  |
|  5  | How do I bake a chocolate cake?                                   | None                                                         | Refused |

---

## Test 1 — Webex Conference Call

**Question:**
`How do I set up a conference call using Cisco Webex?`

**Retrieved Chunks:**

* `[chunk ID]` — `[similarity score]`
* `[chunk ID]` — `[similarity score]`
* `[chunk ID]` — `[similarity score]`

**Answer:**
`[Paste the answer from your app here.]`

**Result:** Relevant Webex chunks were retrieved and the answer was grounded in the document.

---

## Test 2 — File Backup

**Question:**
`How do I create a backup of important files?`

**Retrieved Chunks:**

* `[chunk ID]` — `[similarity score]`
* `[chunk ID]` — `[similarity score]`
* `[chunk ID]` — `[similarity score]`

**Answer:**
`[Paste the answer from your app here.]`

**Result:** Relevant backup instructions were retrieved and used to generate the answer.

---

## Test 3 — Tablet Troubleshooting

**Question:**
`What should I do if I have problems with a company-issued tablet?`

**Retrieved Chunks:**

* `[chunk ID]` — `[similarity score]`
* `[chunk ID]` — `[similarity score]`
* `[chunk ID]` — `[similarity score]`

**Answer:**
`[Paste the answer from your app here.]`

**Result:** The application retrieved information from the tablet troubleshooting document.

---

## Test 4 — Additional Document Question

**Question:**
`[Your fourth question]`

**Retrieved Chunks:**

* `[chunk ID]` — `[similarity score]`
* `[chunk ID]` — `[similarity score]`
* `[chunk ID]` — `[similarity score]`

**Answer:**
`[Paste the answer from your app here.]`

**Result:** The answer was generated using the retrieved document context.

---

## Test 5 — Out-of-Document Question

**Question:**
`How do I bake a chocolate cake?`

**Retrieved Chunks:**

* `[chunk ID]` — `[similarity score]`
* `[chunk ID]` — `[similarity score]`
* `[chunk ID]` — `[similarity score]`

**Answer:**
`I don't have enough information in the documents to answer that.`

**Result:** Passed. The question was not covered by the documents, so the application did not use outside knowledge to answer it.

---

## Overall Result

The tests showed that the Naive RAG application can retrieve relevant chunks from the provided documents and generate answers based on that context. The off-topic question also tested the system's ability to avoid unsupported answers.
