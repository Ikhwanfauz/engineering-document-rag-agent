\# Retrieval Evaluation



\## Version



Version 7B adds reproducible evaluation for semantic retrieval.



\## Purpose



The retrieval evaluation measures whether the retriever finds the expected

manual pages for the answerable questions in the Version 7A evaluation

dataset.



It evaluates:



\- Retrieval hit rate.

\- Expected-page recall.

\- Different top-k values.

\- Different similarity thresholds.

\- Common retrieval failure categories.

\- Machine-readable JSON results.



\## Evaluation Dataset



The evaluator uses:



`evaluation/evaluation\_dataset.json`



The dataset contains four answerable retrieval questions and one

unanswerable question.



Only answerable questions are included in the Version 7B retrieval metrics.

The unanswerable question will be used by later abstention evaluation.



\## Running the Evaluation



Run the evaluator from the project root:



```bat

python -m evaluation.evaluate
