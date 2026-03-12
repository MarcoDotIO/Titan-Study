# Using Dolma to Train a Titans Model

## Purpose

This guide explains how to use the public **Dolma** dataset as a practical pretraining corpus for a **Titans** architecture implementation.

It is written for a proof run or research reproduction attempt where you want a large open corpus without needing private access.

## Important caveat

Dolma is a **good practical substitute**, but it is **not the exact main pretraining dataset used in the Titans paper**.

The paper reports that:

- the **170M, 340M, and 400M** Titans models were trained on **15B tokens sampled from FineWeb-Edu**
- the **760M** model was trained on **30B tokens from FineWeb-Edu**
- one memory-depth ablation was trained on **a subset of The Pile**
- training used a **Llama 2 tokenizer with 32K vocabulary**, **4K training length**, **AdamW**, **learning rate 4e-4**, **cosine annealing**, **batch size 0.5M tokens**, and **weight decay 0.1**

So if you train on Dolma, you should treat the result as a **Titans-on-Dolma reproduction**, not a strict apples-to-apples reproduction of the paper.

## Why Dolma is still a good choice

Dolma is useful because it is public, large, and already filtered/deduplicated enough to support language-model pretraining research.

The current Hugging Face dataset card describes Dolma as:

- a corpus of about **3 trillion tokens**
- built from a mix of **web content, academic publications, code, books, and encyclopedic material**
- available in multiple downloadable versions, including:
  - **v1_7**: about **4.5 TB gzip**
  - **v1_6-sample**: about **16.4 GB** and roughly **10B tokens**

This makes it useful for both:

1. a small proof run on the sample release, and
2. a medium-scale pretraining run on a capped subset of the full release.

## Best Dolma choice for your use case

With about **3 TB** of server storage, the most sensible options are:

### Option A: Fast proof run
Use **Dolma `v1_6-sample`**.

Use this when you want to validate:

- your Titans implementation
- tokenization and packing
- training stability
- memory update logic
- checkpointing and evaluation

This is the safest first run.

### Option B: More serious proof run
Use a **capped subset of Dolma `v1_7`**.

Download only a bounded amount, for example **300 GB to 1 TB**, instead of trying to store the full 4.5 TB compressed release.

Use this when you want:

- more topic diversity than the small sample
- enough data to see whether Titans benefits from longer training
- a realistic pretraining workflow without fully saturating disk

### Option C: Language-only leaning subset
If your goal is to stay closer to the Titans paper's language-model setting, prefer a **text-heavy natural-language subset** of Dolma.

In practice that means biasing toward sources like:

- Common Crawl / Dolma CC
- RefinedWeb
- C4
- books
- encyclopedic material
- academic papers

and optionally reducing the fraction of **code-heavy** shards if your experiment is about general language modeling rather than mixed text+code pretraining.

## Downloading Dolma

### Small sample

```bash
git clone https://huggingface.co/datasets/allenai/dolma && mkdir -p ~/data/dolma && cat dolma/urls/v1_6-sample.txt | xargs -n 1 -P 16 wget -q -P ~/data/dolma
```

### Capped larger run

```bash
git clone https://huggingface.co/datasets/allenai/dolma && mkdir -p ~/data/dolma-proof && wget -c -i dolma/urls/v1_7.txt -P ~/data/dolma-proof --quota=500G
```

Adjust the quota to your storage budget.

## Recommended overall workflow

The clean workflow is:

1. **Download** the Dolma slice you want.
2. **Inspect** the raw records and metadata.
3. **Filter** to the subset you want.
4. **Tokenize** with the same tokenizer family used by the paper.
5. **Pack** tokenized text into fixed-length training sequences.
6. **Train** a Titans variant with paper-aligned hyperparameters where possible.
7. **Evaluate** perplexity and long-context behavior.

## Step 1: Inspect the dataset

Before training, inspect a few records to confirm the fields present in your local files.

Typical things to check:

- where the document text lives
- whether source/provenance metadata is present
- whether documents are already normalized
- whether some shards are corrupted or incomplete

A minimal Python inspection pattern is:

```python
from datasets import load_dataset

# Adjust to your local layout.
# In many cases you can point HF datasets to the local clone/data directory.
ds = load_dataset("allenai/dolma", split="train")
print(ds[0])
```

If the full dataset is too large to index this way, inspect shard files directly and build a streaming pipeline.

## Step 2: Decide the data mixture

This is the most important choice after architecture.

### If the goal is paper alignment
Use a **natural-language-only** subset.

Reason:

- Titans' main language experiments were trained on **FineWeb-Edu**, which is an educational/web-text corpus rather than a mixed code corpus.
- Dolma contains more heterogeneous sources, including code.
- A strongly mixed corpus may change both perplexity behavior and the type of long-range dependencies the model sees.

### If the goal is systems validation
Use the full mixed Dolma slice.

Reason:

- for a proof run, you mainly need enough real text to exercise the pipeline
- Titans' memory mechanism should still be testable on mixed-domain data

## Step 3: Tokenize in a paper-aligned way

To stay close to the paper, use a **Llama 2 tokenizer with 32K vocabulary**.

That is the tokenizer setup explicitly stated in the paper.

### Recommended tokenization rules

- append an end-of-document token between documents
- preserve document boundaries before packing
- do not randomly splice arbitrary text without separators
- log the total token count after tokenization
- build train/validation splits at the **document level**, not at the token slice level

### Why this matters for Titans

Titans is explicitly designed around:

- local short-term context
- long-term memory updates across sequence flow
- surprise-driven memorization and forgetting

If you destroy document structure too aggressively, you make the long-term memory problem less meaningful.

## Step 4: Pack into fixed-length sequences

The paper states a **training length of 4K tokens**.

For a first implementation, pack your token stream into **4096-token** sequences.

A good default is:

- concatenate tokenized documents with EOS separators
- pack into contiguous 4096-token blocks
- keep a small held-out validation set

### Do not do this for the first run

Avoid starting with:

- 16K or longer packed sequences
- dynamic packing plus multiple curriculum changes
- aggressive source mixing experiments

That makes debugging impossible.

## Step 5: Choose the Titans variant

The paper evaluates:

- **LMM**: neural memory module alone
- **MAC**: Memory as Context
- **MAG**: Memory as Gate
- **MAL**: Memory as Layer

For a first serious reproduction attempt:

### Best default: MAC
Choose **MAC** if your goal is strongest paper-style long-context performance.

The paper reports that MAC and MAG are close on language modeling and reasoning, while **MAC performs better on long-context NIAH tasks**.

### Simpler alternative: LMM
Choose **LMM** if you want to validate the long-term memory mechanism in isolation before integrating a full hybrid architecture.

### Efficiency-oriented alternative: MAL
Choose **MAL** only if you prioritize throughput more than matching the strongest long-context behavior.

## Step 6: Use paper-aligned optimizer settings

The paper's training setup states:

- **AdamW** optimizer
- **learning rate = 4e-4**
- **cosine annealing** schedule
- **global batch size = 0.5M tokens**
- **weight decay = 0.1**
- **sequence length = 4K**

For a proof run, you usually cannot keep the exact same batch size unless you have substantial hardware.

So the correct adaptation is:

- keep the **optimizer family** the same
- keep the **learning-rate scale** near the paper's setup
- use **gradient accumulation** to approximate a larger token batch
- keep **weight decay = 0.1** unless training becomes clearly unstable

## Step 7: Set a realistic proof-run scale

Do not jump directly to the paper's 15B-token and 30B-token training budgets.

A practical ladder is:

### Stage 1: Pipeline validation

- dataset: `v1_6-sample`
- target processed tokens: whatever your tokenization pipeline yields from the sample
- model: small Titans, such as roughly **100M to 200M** parameters
- objective: verify loss decreases, no NaNs, checkpoints resume correctly

### Stage 2: Medium proof run

- dataset: capped `v1_7` subset
- target: a few **billion** tokens after tokenization, depending on your hardware budget
- model: roughly **200M to 400M** parameters
- objective: observe whether long-term memory helps beyond ordinary short-context modeling

### Stage 3: Paper-style run

- dataset: larger curated text-only Dolma subset or switch to FineWeb-Edu for closer reproduction
- token budget: approach the paper's **15B-token** regime
- model: 340M to 400M class

## Step 8: Evaluate the right thing

A common failure mode is to train Titans like a normal LM and only look at training loss.

That is insufficient.

Titans is supposed to help with **memory over long context**, not merely reduce next-token loss on short windows.

So you should evaluate at least:

- validation perplexity
- long-context retrieval tests
- needle-in-a-haystack style evaluation
- ablations of memory depth or memory disabling

The paper specifically uses:

- language modeling and commonsense reasoning benchmarks
- **RULER Single NIAH** for long-context evaluation
- time-series datasets
- genomics benchmarks

For a text-only reproduction, the most relevant checks are:

1. perplexity on held-out text
2. a long-context retrieval benchmark
3. comparison against the same core model with memory disabled or replaced

## Data pipeline recommendations specific to Titans

### 1. Preserve continuity
Titans is designed to exploit temporal/document flow. Prefer contiguous text over randomly shuffled sentence fragments.

### 2. Split by document hash
Validation leakage is easy to create in large web corpora. Split by document identity, not by packed chunk.

### 3. Keep EOS boundaries
Memory and surprise dynamics become harder to interpret if documents are merged with no explicit delimiter.

### 4. Track token counts, not bytes
Compressed download size is not the same as trainable token count.

### 5. Log source mixture
If Dolma source metadata is available in your pipeline, log the fraction of web, books, code, papers, and encyclopedic text. Otherwise you will not know what you actually trained on.

### 6. Start with fixed 4K length
First reproduce the paper's basic training regime. Only then test longer windows or chunk-level modifications.

## What I would recommend for your setup

Given your storage budget, the most practical plan is:

### Recommended plan

1. Run **one complete training-and-eval cycle** on `v1_6-sample`.
2. After the pipeline works, run a **500 GB to 1 TB capped `v1_7` download**.
3. Filter that second run toward **natural-language-heavy** sources.
4. Train a **small-to-mid-size MAC model** with:
   - Llama 2 tokenizer (32K)
   - 4K packed sequences
   - AdamW
   - lr 4e-4 starting point
   - cosine schedule
   - gradient accumulation to simulate larger token batches
5. Compare against a baseline without Titans memory or with reduced memory depth.

That is the highest-value sequence of experiments for a serious proof run.

## Minimal implementation checklist

- [ ] Dolma shard download works
- [ ] text extraction verified on sample records
- [ ] tokenizer produces stable counts
- [ ] EOS handling implemented
- [ ] document-level split implemented
- [ ] 4096-token packing implemented
- [ ] Titans model forward pass verified
- [ ] memory update path verified
- [ ] gradient accumulation verified
- [ ] checkpoint resume verified
- [ ] validation perplexity script verified
- [ ] long-context evaluation script verified

## Bottom line

Use **Dolma** as a practical public training corpus, but be explicit that it is a **substitute corpus** for Titans rather than the paper's exact main dataset.

For your setup, the best path is:

- **first**: `v1_6-sample` for a complete proof run
- **second**: a **capped `v1_7` subset** for a more meaningful experiment
- **third**: keep the **paper's tokenizer, 4K sequence length, and optimizer recipe** so the architecture comparison remains interpretable

## References

- Behrouz, A., Zhong, P., & Mirrokni, V. *Titans: Learning to Memorize at Test Time*.
- Hugging Face dataset card: *allenai/dolma*.
