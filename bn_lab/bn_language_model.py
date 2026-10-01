"""First- and second-order count-based autoregressive language models."""
from collections import Counter, defaultdict
import random
from pathlib import Path

SENTENCES = [
    "the cat sat on the mat", "the cat sat on the rug",
    "the dog sat on the mat", "the dog ran to the park",
    "the cat ran to the park", "the dog sat on the rug",
]
START, END = "<START>", "<END>"
DATA = [[START, *s.lower().split(), END] for s in SENTENCES]


class NGram:
    def __init__(self, order=1):
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        self.order = order
        self.counts = defaultdict(Counter)
        for raw_sentence in SENTENCES:
            sentence = [START] * order + raw_sentence.lower().split() + [END]
            for i in range(order, len(sentence)):
                context = tuple(sentence[i-order:i])
                self.counts[context][sentence[i]] += 1
        self.probabilities = {
            ctx: {word: n / sum(counter.values()) for word, n in counter.items()}
            for ctx, counter in self.counts.items()
        }

    def distribution(self, context):
        context = tuple(context) if isinstance(context, (tuple, list)) else (context,)
        context = context[-self.order:]
        return self.probabilities.get(context, {})

    def predict(self, context):
        d = self.distribution(context)
        return max(d, key=d.get) if d else None

    def generate(self, rng, greedy=False, max_tokens=30):
        history = [START] * self.order
        result = []
        for _ in range(max_tokens):
            d = self.distribution(history[-self.order:])
            if not d:
                break
            if greedy:
                token = max(d, key=d.get)
            else:
                words, weights = zip(*d.items())
                token = rng.choices(words, weights=weights, k=1)[0]
            if token == END:
                break
            result.append(token)
            history.append(token)
        return " ".join(result)


def main():
    first = NGram(1)
    for w in ("the", "cat", "dog", "sat", "ran"):
        print(w, first.distribution(w))
    print("normalization", {c: round(sum(p.values()), 12) for c, p in first.probabilities.items()})
    print("predictions", {w: first.predict(w) for w in ("<START>", "the", "cat", "dog", "sat", "ran")})
    rng = random.Random(2026)
    samples = [first.generate(rng) for _ in range(20)]
    Path(__file__).with_name("bn_generated_sentences.txt").write_text(
        "\n".join(samples) + "\n", encoding="utf-8"
    )
    print("sampled 20")
    for sample in samples:
        print(sample)
    print("greedy x5", [first.generate(random.Random(i), greedy=True) for i in range(5)])
    print("sampling x5", [first.generate(random.Random(100+i)) for i in range(5)])
    second = NGram(2)
    print("second order distinct contexts", len(second.counts))
    print("second order sample", [second.generate(random.Random(i)) for i in range(5)])


if __name__ == "__main__":
    main()
