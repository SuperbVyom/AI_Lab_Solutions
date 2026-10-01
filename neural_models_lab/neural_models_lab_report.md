# Neural Models: XOR, Activations, and Output Layers

## Task 1: specification and linear separability

Input space is `X = {(0,0),(0,1),(1,0),(1,1)}` and output is `{0,1}`. The labels in that order are `0,1,1,0`. The positive examples are opposite corners of the square, as are the negative examples, so one straight line cannot put both positive corners on one side and both negative corners on the other. A single affine map followed by sigmoid has a linear decision boundary and therefore cannot represent XOR exactly, regardless of training duration.

## Task 2: model design and success criteria

The binary design is 2 inputs -> 2 hidden units with `tanh` -> one output logit, trained with `BCEWithLogitsLoss` and Adam. The hidden nonlinearity lets the network form a non-linear decision boundary; adding affine layers alone still composes to an affine transformation. Sigmoid probabilities with binary cross-entropy fit a yes/no target, and `BCEWithLogitsLoss` combines them stably. Success should be judged by low final loss, all four thresholded labels correct, and gradients that are finite/nonzero early in training and lead to improving loss. Repeated seeds provide an additional robustness check.

The hidden units have no separate labels. Backpropagation computes how changing each unit affects the task loss, so gradient descent discovers useful intermediate features through the final XOR objective.

## Task 3: prompt and code

**Prompt:** “Generate concise CPU PyTorch code for the four XOR examples `(0,0)->0`, `(0,1)->1`, `(1,0)->1`, `(1,1)->0`. Use exactly a 2-2-1 MLP, a selectable sigmoid/tanh/ReLU hidden activation, one output logit with `BCEWithLogitsLoss`, random initialization, a fixed seed, full-batch Adam training, and print initial/final loss, four probabilities and labels, and an early first-layer gradient norm. Also run a zero-initialized symmetry control and a three-logit extension using class labels 0,1,1,2 and cross-entropy. Keep the scientific task unchanged.”

`neural_xor.py` implements the binary and three-class experiments. The forward calculation is `Net.forward`; scalar losses use the corresponding PyTorch loss functions; `loss.backward()` invokes reverse-mode automatic differentiation; `opt.step()` updates parameters. The first-layer gradient tensor is inspected through `model.fc1.weight.grad`.

## Task 4: analysis and validation criteria

The lab asks for measured losses, predictions, gradients, symmetry trajectories, and an activation comparison. **Those PyTorch measurements were not run in this environment because neither PyTorch nor NumPy is installed.** The source is supplied, but its numerical outputs are intentionally not fabricated. Run `python neural_xor.py` in an environment with PyTorch to populate the measured values. The expected structural result of the zero-initialization control is that the two hidden units remain identical: with identical parameters and inputs they produce identical activations and receive identical gradients, so gradient-based updates preserve their symmetry. The network cannot use the two units to specialize into different features under this initialization.

When using a mean batch loss, the parameter gradient is the average of the four per-example gradients by linearity of differentiation. `parameter.grad` is the derivative of that scalar batch loss with respect to the parameter. A merely nonzero gradient is not sufficient evidence of useful learning; declining loss and correct predictions show that the update direction helps the objective.

| Hidden activation | Final loss | 4/4 correct | Early `||grad W1||` |
|---|---|---|---|
| Sigmoid | not measured here | not measured here | not measured here |
| Tanh | not measured here | not measured here | not measured here |
| ReLU | not measured here | not measured here | not measured here |

Sigmoid saturation yields a small derivative when pre-activations have large magnitude, while ReLU yields zero derivative for negative pre-activations and one for positive pre-activations. Inspecting pre-activations and activations distinguishes these cases. Tanh can also saturate. Differences in final fit for four points are observations about this setup, not evidence of a universally best activation.

## Task 5: multiclass extension

The final affine matrix has shape `3 x 2` (plus three biases), and each example has three logits. Softmax outputs `p_i = exp(z_i)/sum_j exp(z_j)`, which sum to one by normalization. For one-hot target `y`, cross-entropy is `-sum_i y_i log(p_i)`; differentiating with respect to a logit gives `p_i-y_i`. The source uses the stable fused `CrossEntropyLoss` and prints all probability vectors, a row sum, and the effect of adding 100 to the logits. Numerically expected row sums are approximately one and common-shifted logits leave softmax unchanged, but measured values require the missing runtime.

The three-class labels are `(0,0)->0`, `(0,1)->1`, `(1,0)->1`, `(1,1)->2`. The softmax and cross-entropy mathematics stays the same with a large vocabulary; the output dimension and surrounding model, data volume, and compute costs grow substantially.

## Reflection questions

1. XOR shows that nonlinearity is essential: affine-only depth remains affine and cannot separate XOR, while nonlinear hidden units can create a suitable representation.
2. Useful learning is evidenced by falling loss and correct predictions, not just nonzero gradients. The gradient should be finite and updates should improve the objective.
3. With identical zero weights, hidden units are exchangeable and receive the same gradient. They remain identical, so they cannot learn distinct feature detectors; zero output weights can also initially block a useful gradient into the hidden layer.
4. Saturated sigmoid/tanh units have small derivatives; inactive ReLU units have exactly zero derivatives. The scientific explanation follows from activation derivatives; engineering outcomes such as final loss also depend on seed, initialization, and optimizer settings.
5. The output/loss pairing must match the target semantics: sigmoid plus binary cross-entropy for independent binary prediction, softmax plus categorical cross-entropy for one-of-K classification. A mismatched output gives the wrong probability interpretation or learning signal.
6. The LLM can rapidly produce model/optimizer boilerplate and expose gradients. Human review is essential to catch architecture drift, wrong labels, output/loss mismatches, or claims of success unsupported by loss and prediction checks.
7. Keep inexpensive checks such as loss trends, prediction metrics, gradient finiteness/norms, shape assertions, and sampled gradient checks. Exhaustive finite differences become costly as parameter count grows; use them only on tiny models or selected parameters.

## Execution environment note

`neural_xor.py` requires PyTorch as requested by the lab. It remains a reproducible runnable deliverable, but this folder does not contain verified PyTorch measurements because this workstation's available Python runtimes do not provide PyTorch or NumPy.
