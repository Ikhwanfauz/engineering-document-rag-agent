# Configuration Comparison

## Version

Version 7D compares the evaluated retrieval and language-model configurations and selects the final supported configuration for the Engineering Document RAG Agent.

## Evaluation setup

The evaluation uses:

- Evaluation dataset: `evaluation/evaluation_dataset.json`
- Source document: `e-Series_Service_Manual_en.pdf`
- Five evaluation questions
- Four answerable questions
- One unanswerable question
- Persistent ChromaDB vector store
- `all-MiniLM-L6-v2` embeddings

The same evaluation questions and retrieval settings were used for both language models.

## Retrieval configuration comparison

Version 7B evaluated:

- `top_k`: 1, 3, and 5
- Minimum similarity threshold: none, 0.50, 0.60, and 0.70

### Retrieval results

| Top K | Threshold | Hit rate | Expected-page recall |
|---:|---:|---:|---:|
| 1 | None | 75.00% | 75.00% |
| 1 | 0.50 | 75.00% | 75.00% |
| 1 | 0.60 | 75.00% | 75.00% |
| 1 | 0.70 | 25.00% | 25.00% |
| 3 | None | 100.00% | 100.00% |
| 3 | 0.50 | 100.00% | 100.00% |
| 3 | 0.60 | 100.00% | 100.00% |
| 3 | 0.70 | 100.00% | 100.00% |
| 5 | None | 100.00% | 100.00% |
| 5 | 0.50 | 100.00% | 100.00% |
| 5 | 0.60 | 100.00% | 100.00% |
| 5 | 0.70 | 100.00% | 100.00% |

### Selected retrieval configuration

The selected retrieval configuration is:

- `top_k=3`
- Minimum similarity threshold: `0.60`

This configuration retrieved all expected pages in the current evaluation dataset while returning fewer chunks than `top_k=5`. The threshold also filters weaker evidence before answer generation.

## Language-model comparison

Version 7D compared:

- `qwen3:8b`
- `llama3.2`

Both models used the same evaluation dataset and selected retrieval configuration.

### Answer-evaluation results

| Metric | `qwen3:8b` | `llama3.2` |
|---|---:|---:|
| Abstention accuracy | 100.00% | 100.00% |
| Citation correctness | 75.00% | 75.00% |
| Answer-point coverage | 56.95% | 62.35% |
| Evidence grounding score | 86.38% | 81.27% |
| Average response latency | 4.25 seconds | 4.32 seconds |
| Total elapsed time | 21.32 seconds | 21.70 seconds |

### Qualitative comparison

Both models correctly answered the four answerable questions and abstained from the unanswerable question.

`llama3.2` achieved higher answer-point coverage and produced clear, structured answers. However, it sometimes added information that was not required by the question. For example, it added wire-removal instructions to the clamp-support answer and connected blue-lid removal with preventing the joint from falling.

`qwen3:8b` produced more focused answers and achieved the higher overall evidence-grounding score. It generally answered the exact question without adding as much unnecessary or potentially misleading information.

## Final supported configuration

The final supported configuration is:

- Embedding model: `all-MiniLM-L6-v2`
- Vector store: ChromaDB
- Retrieval count: `top_k=3`
- Minimum similarity threshold: `0.60`
- Language model: `qwen3:8b`

`qwen3:8b` remains the selected language model because evidence grounding and focused answers are especially important for an engineering service-manual assistant. Its latency and citation correctness were also comparable to `llama3.2`.

## Limitations

- The current evaluation dataset contains only five questions from one service manual.
- The results do not prove that the selected configuration will perform equally well on other documents or question types.
- Answer-point coverage uses wording similarity and can underrate correct answers that use different phrasing.
- Citation correctness penalizes additional cited pages even when those pages contain related information.
- Generated answers and latency can vary between runs.
- The first evaluated question may have higher latency because models and embedding components are still loading.
- The evaluation currently compares only two local language models.
- Human review is still required when interpreting small differences between automatic metric scores.

## Improvement opportunities

Future work should:

- Expand the evaluation dataset with more documents and question types.
- Add more answerable and unanswerable cases.
- Improve semantic answer-point scoring.
- Distinguish harmful citations from related but unnecessary citations.
- Run repeated evaluations to measure variability.
- Compare additional language models and retrieval strategies.
- Add safety and prompt-injection evaluations.
- Track evaluation results across future versions to detect regressions.

## Reproducing the evaluation

Run the retrieval evaluation:

```bat
python -m evaluation.evaluate