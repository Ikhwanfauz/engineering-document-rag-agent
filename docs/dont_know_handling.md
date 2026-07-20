# Don't-Know Handling

## Version

Version 3B adds controlled abstention when the indexed documents do not provide sufficiently relevant evidence.

## Why abstention is required

Vector search always returns the nearest available chunks, even when the document does not contain the answer.

For example, a question about car wheel-nut torque retrieved a robot-manual section titled `Torque Values`. The wording was similar, but the evidence did not answer the question.

Without an evidence gate, this weak result could be sent to the LLM and produce an unsupported technical answer.

## Evidence-gating behavior

The pipeline uses a minimum cosine similarity of `0.60`.

The processing flow is:

1. Retrieve the nearest document chunks.
2. Remove chunks with similarity below `0.60`.
3. If no chunks remain, return a controlled abstention.
4. Skip LLM generation and return no citations.
5. If accepted evidence remains, send only those chunks to the LLM.

The controlled response is:

```text
I don't know based on the uploaded documents.