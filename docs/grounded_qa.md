\# Grounded Question Answering



\## Version



Version 3A adds grounded answer generation with page-level citations.



\## Components



\- `src/llm\_provider.py` provides a configurable Ollama LLM interface.

\- `src/rag\_pipeline.py` retrieves evidence, builds the grounded prompt, generates an answer, and validates mandatory language.

\- `src/citation\_manager.py` creates document, page, and excerpt citations.

\- `scripts/ask\_manual.py` provides the end-to-end command-line interface.



\## Grounding behavior



The LLM receives only the user question and retrieved evidence. The prompt requires it to:



\- use directly relevant evidence only;

\- preserve explicit procedural order;

\- preserve mandatory actions;

\- avoid inferred explanations and unrelated steps;

\- provide concise answers.



When evidence contains `MANDATORY ACTION`, the pipeline requires the answer to use mandatory wording and retries once if the response softens the instruction.



\## Local models



Ollama is used so answer generation remains local and configurable.



`llama3.2` established the initial working baseline. `qwen3:8b` followed procedural instructions more reliably and is the preferred tested model for the engineering manual.



Qwen reasoning is disabled and output length is bounded to avoid unnecessarily long generation.



\## Example



```bash

python -m scripts.ask\_manual "How should the joint be supported when removing the clamp?" --top-k 3 --llm-model qwen3:8b

```



\## Validation



Version 3A was validated against the Universal Robots e-Series service manual. Tests covered:



\- grounded answers and citations;

\- empty questions and missing evidence;

\- mandatory-action correction and rejection;

\- configurable Ollama generation;

\- citation formatting;

\- contaminated PDF footer cleanup;

\- real retrieval-to-answer execution.



The automated suite contains 51 passing tests.



\## Known limitation



Small local models may omit a precaution from broad multi-page evidence or include a related but unrequested detail. Retrieval and evidence cleaning were verified independently. Exhaustive evidence-coverage validation and insufficient-evidence abstention remain later improvements.



\## Next checkpoint



Version 3B adds similarity thresholds and controlled "I don't know based on the uploaded documents" behavior.
