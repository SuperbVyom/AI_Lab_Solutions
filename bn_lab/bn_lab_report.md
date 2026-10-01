# Bayesian Networks and Autoregressive Language Models

## Questions 1-3: model and CPT

The chain rule is useful for generation because it turns a joint sequence probability into a sequence of next-token choices. After sampling each token from its conditional distribution, the model can append it and continue. In the first-order network, the conditional independence assumption is `X_t ⫫ (X_1,...,X_{t-2}) | X_{t-1}`: given the immediately previous word, earlier words do not affect the next-word distribution.

The provided six sentences are lowercased, tokenized by whitespace, and wrapped in `<START>`/`<END>`. The empirical first-order distributions are:

| Current token | Next token distribution |
|---|---|
| `<START>` | the: 1 |
| the | cat: 1/4, dog: 1/4, mat: 1/6, rug: 1/6, park: 1/6 |
| cat | sat: 2/3, ran: 1/3 |
| dog | sat: 2/3, ran: 1/3 |
| sat | on: 1 |
| ran | to: 1 |

Other observed transitions: `on -> the (1)`, `to -> the (1)`, `mat -> <END> (1)`, `rug -> <END> (1)`, `park -> <END> (1)`. Any transition not listed has probability zero, including `cat -> dog`, `sat -> ran`, and `park -> cat`.

## Questions 4-9: implementation and validation

`bn_language_model.py` stores transition counts as `Counter`s keyed by context tuples. It divides each count by the total count for that context to form the CPT. Greedy prediction returns the maximum-probability token; sampling draws according to the normalized probabilities. Greedy choice is deterministic and can suppress alternatives, while sampling reflects the full distribution. An unseen context has no outgoing distribution; generation halts cleanly (a system with smoothing/backoff could use a fallback instead).

A row total of 0.87 indicates a normalization/counting bug, such as omitting an observed successor from the denominator or dividing by the wrong total. All observed context totals in the run are 1.0. For five word contexts, the most probable results are: `the -> cat` (tie with dog, resolved by data order), `cat -> sat`, `dog -> sat`, `sat -> on`, `ran -> to`. These are maximum-likelihood corpus predictions, not universal grammar judgments; the model has only six training examples.

Twenty seeded sampled generations (saved by the executable output) include short corpus phrases and mixed sequences such as `the dog sat on the cat ran to the dog ran to the park`. Such mixtures are expected for a first-order model because it forgets longer context. Five greedy runs repeat the same mode-limited output and continue `the cat sat on the cat sat...` until the safety length cap; sampling has more variation and can terminate at `<END>`.

## Questions 10-12: generation and second order

Sampling produced more variation because every transition with nonzero probability remains possible; greedy generation repeatedly selects the same maximum and can enter loops. The second-order graph has two parent variables for each token, `X_{t-2}` and `X_{t-1}`, instead of the single edge from `X_{t-1}`. Its table is indexed by observed pairs and stores counts of triples. It offers a longer context but requires more data: possible contexts grow roughly with vocabulary-size squared, leaving more rows unseen and estimates sparse. Context can improve prediction by disambiguating words that share the same immediate predecessor, while sparse data makes those conditional estimates less reliable.

On the six-sentence corpus, the program counts 15 distinct second-order contexts. The model distinguishes `the cat -> sat/ran` and `the dog -> sat/ran`; because one preceding pair can still have multiple continuations, it remains stochastic. First order has 11 observed contexts and 17 distinct observed context-successor parameters; second order has 15 observed contexts and 19 observed context-successor parameters (distinct outcomes with nonzero empirical probability). A context outside the training set has zero outgoing probability mass unless smoothing/backoff is added.

## Modern language models and LLM reflection

A neural autoregressive language model still estimates `P(X_t | X_1,...,X_{t-1})` and factorizes the joint sequence probability by the chain rule. The CPTs and count-based learning here are replaced by neural-network weights and gradient-based training; generation still samples or selects a next token. The longer learned context and parameter sharing let neural models generalize across contexts, but do not change the autoregressive factorization.

Specifying the behavior before prompting makes the intended representation and computation testable. The code can be inspected to ensure it counts adjacent tokens and normalizes per context, then tested using row sums and known predictions. For example, a generated version that forgot to add `<END>` transitions would produce endless sentences; adding end-token counting fixes this and is checked by the generation cap and observed stop probabilities. The model is the probabilistic specification; its implementation is the data structures and routines that realize it. Thinking in Bayesian-network terms makes dependencies, factorization, conditional meaning, generation, independence assumptions, context growth, and invariants explicit.

## Exact implementation prompt

“Implement a first-order autoregressive model from the six provided tokenized sentences with `<START>` and `<END>` markers. Count every consecutive-token transition including the end marker, report normalized next-token distributions, deterministic argmax predictions, seeded sampling, and at least twenty generated samples. Check every row sums to one. Then implement a second-order model based on token triples and compare its observed contexts and generation. Use standard Python only and stop generation at `<END>` or a finite token cap.”

## Deliverables

The runnable implementation is `bn_language_model.py`. Its output includes the requested conditional tables, normalization totals, predictions, 20 samples, greedy/sampling comparison sets, and second-order examples.
