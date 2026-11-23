# Where to adapt an ASR foundation model for dysarthric speech

Code and aggregate results for the CANDOR study of **encoder depth and adapter placement** in dysarthric
ASR: layer-wise probing of the Qwen3-ASR-1.7B audio encoder, and a four-band LoRA placement sweep
replicated across two corpora.

**Paper:** _publication URL / DOI to be added on acceptance._
**Programme:** [CANDOR](https://smflabs.org) — Cross-lingual Adaptive Neural Dysarthric Speech Output
Recognition, The Scott-Morgan Foundation / SMF Labs.

---

## The question

Parameter-efficient adaptation of a speech foundation model to dysarthric speech forces a design choice
that is rarely justified: **which encoder layers should the adapter go in?**

Prior work on a 12-block Whisper-small encoder probed it layer by layer, found phoneme identity matures
mid-stack, and showed adapters placed in that middle band recover most of the benefit of adapting the
whole encoder. This study asks whether that is a fact about *dysarthric speech* or a fact about *that
encoder*, and answers it along three axes:

1. **Hold the corpus constant, swap the encoder.** Same Mandarin corpus (CDSD), 24-block Qwen3-ASR audio
   encoder whose decoder is a language model rather than a Whisper-style text decoder.
2. **Hold the encoder constant, swap the corpus.** Identical sweep on UASpeech — English, isolated words,
   clinical severity labels rather than CER-derived ones.
3. **Re-run the prior architecture ourselves, rather than citing it.** Whisper-small probed *and* swept
   under this repository's own corpus, split, integrity gate, recipe and scoring — and the probe
   additionally at matched training volume and under both padding treatments.

Axis 3 was added in August 2026 and **changed the paper's central claim**; see the revision note below.

## Headline findings

> **Revision note (2026-08-20).** An earlier version of this work claimed the two architectures' encoder
> depth profiles differ, and proposed a "knee rule" reconciling our results with the prior work. Both
> rested on the Whisper-small depth profile *as published elsewhere*. Having since measured it here — and
> at matched probe-training volume — we find the two encoders mature at essentially the same relative
> depth, and the knee rule makes a **falsified** prediction on Whisper-small. Those claims are withdrawn.
> The adaptation results below are unchanged.

**Adapter placement is architecture-specific.** Holding corpus, split, gate, recipe, parameter budget and
seed discipline constant and swapping only the encoder changes which band wins.

| Band | Qwen3-ASR blocks | Params | CDSD CER | UASpeech CER | Whisper-small blocks | CDSD CER |
|---|---|---|---|---|---|---|
| base (no adaptation) | — | 0 | 32.06 | 41.95 | — | 77.57 |
| `lower` | 1–8 | 4.19 M | 35.52 ± 1.15 | 52.02 | 1–4 | 56.28 ± 0.07 |
| `middle` | 9–16 | 4.19 M | 30.71 ± 0.31 | 37.61 | 5–8 | 55.16 ± 0.13 |
| `upper` | 17–24 | 4.19 M | **29.88 ± 0.36** | 34.19 | 9–12 | 56.95 ± 0.08 |
| `all` | 1–24 | 12.58 M | 31.19 ± 0.37 | **32.66** | 1–12 | **53.56 ± 0.01** |

**Every adaptation cell is a three-seed mean ± SD (seeds 42/43/44).** Note how much more seed-stable the
Whisper × CDSD cell is than the Qwen one (SD 0.01–0.13 against 0.31–1.15).

`upper` wins on Qwen3-ASR — best band in all three seeds — and ranks **worst of the four** on
Whisper-small, where `all` wins with a resolved margin (−3.49 pp, CI [−5.12, −1.92]). The `all` vs `upper` contrast is resolved in *opposite
directions* on the two architectures — **on CDSD only**. Running the same contrast in all four cells of a
model × corpus design (added August 2026) shows the reversal is specific to that corpus: on UASpeech
neither model favours `upper`. A parameter-matched 2×2 rules out total budget as the
explanation on the Qwen3-ASR side, but does not resolve placement itself — see the 2×2 note below.

> **2×2 note (2026-08-28).** The placement × capacity 2×2 was first run at a single seed with LoRA α
> pinned at 128 for every rank, so effective scaling α/r differed 3× within the matched pair. Rebuilt with
> **α = 2r in all four cells across three seeds**:
>
> | contrast | Δ (pp) | seed SD | 95% CI | |
> |---|---|---|---|---|
> | `upper` r192 − `all` r64 (exact 12.58 M match) | −1.28 | 0.37 | [−2.36, **+0.08**] | spans zero |
> | `upper` r64 − `all` r21 (~4.2 M match) | −0.78 | 0.29 | [−1.55, +0.39] | spans zero |
> | `upper` r192 − `upper` r64 (capacity, placement fixed) | +0.03 | 0.38 | [−1.33, +0.73] | spans zero |
> | `all` r64 − `all` r21 (capacity, placement fixed) | +0.53 | 0.14 | [−0.17, +1.14] | spans zero |
>
> **The budget-matched contrast no longer resolves** (it was −1.55, CI [−2.41, −0.57]). Worth separating
> *why*: matching α/r moved each cell by only ~0.1 pp and at seed 42 made the effect *larger* (−1.67 vs
> −1.55). What opened the interval was training stochasticity — −1.67 / −0.94 / −1.24 across seeds. The
> scaling defect was real but empirically inert; the single seed was the problem, here as in the band
> sweep.
>
> **What the 2×2 does establish is negative and useful: added capacity helps neither placement.** Tripling
> rank gives +0.03 within `upper` and +0.53 within `all` — so `all` is not losing for want of parameters.
> The resolved placement claim is the equal-rank contrast (−1.31, CI [−2.16, −0.25]), not this one.
> Reproduce with `scripts/alpha_matched_2x2.sh` → `scripts/alpha_matched_contrasts.py`.

> **Estimator note (2026-08-30).** With only 6 CDSD / 5 UASpeech test speakers, no single interval
> estimator should carry a conclusion alone. Every headline contrast is therefore reported three ways:
> the speaker × seed bootstrap on **micro** CER (pools edit counts — the field-standard ASR metric), the
> same bootstrap on **macro** CER (speakers weighted equally), and an **exact** paired sign-flip p-value
> enumerating all 2⁶ = 64 (or 2⁵ = 32) sign assignments.
>
> | cell | contrast | micro | macro | exact p |
> |---|---|---|---|---|
> | Whisper × CDSD | `all` − `upper` | **−3.40** resolved | **−3.63** resolved | **0.031** |
> | Qwen × CDSD | `upper` − `all` | **−1.31** resolved | −0.98 **spans zero** | 0.156 |
> | Qwen × CDSD | `upper` − `lower` | **−5.64** resolved | **−5.32** resolved | **0.031** |
>
> **Three of four agree; the exception is the Qwen headline placement contrast.** It resolves under
> micro CER and not under speaker-equal weighting — five of six speakers favour `upper`, and the one that
> does not contributes little enough text for micro-averaging to down-weight it. We state that dependency
> rather than hide it. The Whisper half of the cross-model reversal resolves under every estimator,
> unanimously across speakers.
>
> **On multiplicity:** Holm over a three-hypothesis family returns nothing significant, *including* the
> unanimous 5.6 pp contrast — because with 6 speakers the smallest attainable exact p is 0.031, so Holm's
> floor is 0.094. That is a property of the design, not of the data, and we report the floor rather than a
> verdict determined by it. Six speakers is the binding constraint on this study.
> Reproduce: `scripts/exact_signflip.py`, `scripts/macro_bootstrap_check.py`.

**But the depth profile does not explain it.** Measured here rather than cited, Whisper-small's
phoneme-error minimum sits at layer 10 of 12 — **83% of relative depth**, not the 50% the published curve
implies — against 83–92% for Qwen3-ASR. Two encoders that mature at the same relative depth prefer
different bands, and all three candidate probe-to-placement rules fail:

**Selection regret** — how much CER each heuristic actually costs, rather than whether it names the
argmin (`scripts/regret_and_holm.py`): *steepest improvement* never picks the best band and costs up to
**19.36 pp**; *lowest band-mean PER* and *knee* each pick it in 1 of 4 cells at a mean cost of **2.23 pp**.
Following the best available heuristic gives up ~2 pp CER on average against a four-run sweep.

| Rule | Qwen3-ASR | Whisper-small |
|---|---|---|
| adapt where PER improves steepest | **fails** — steepest band is the worst target | — |
| adapt where band-mean PER is lowest | holds | **fails** — lowest-PER band ranks worst |
| adapt where the knee sits (our own rule) | holds by construction | **falsified** — predicts `upper`; `all` wins |

**So the practical recommendation changed: sweep, don't probe.** A four-band sweep costs a few adaptation
runs and answers the question directly. Probing remains useful for what it measures — where information
is, and how reliably it can be read out at each depth — but it does not select the band.

**What replicates across corpora** (intervals excluding zero on *both*): `upper` > `lower` and
`middle` > `lower`. `upper` > `middle` resolves on UASpeech but **not** on CDSD once training seeds are
resampled — it points the same way in all three seeds, so we report it as directional rather than
established. No contrast is resolved in opposite directions on the two corpora.

> **Seed note (2026-08-27).** The CDSD sweep was originally one training run per band, with intervals
> from a speaker-only bootstrap — an estimator that says which six speakers were held out but nothing
> about which initialisation the adapter got. With headline contrasts around 1.5 pp that mattered. All
> four bands were retrained at seeds 42/43/44 (~35 GPU-hours) and every CDSD interval now comes from a
> **speaker × seed** bootstrap, contrasts formed within a seed.
>
> **One conclusion did not survive.** `upper` − `middle` was −1.52, CI [−2.27, −0.54], excluding zero;
> it is now **−0.83, CI [−2.06, +1.86], including zero.**
>
> **The result the main claim rests on did.** `upper` − `all` moves −1.47 → **−1.31, CI [−2.16, −0.25]**,
> still resolved, agreeing with the parameter-matched 2×2 (−1.26 at equal rank). The seed sensitivities
> differ by 3.7× — seed SD 0.16 for `upper` − `all` against 0.59 for `upper` − `middle` — so the contrast
> with independent corroboration is also the stable one, and the one that had none is the one that
> dissolves. **General caution, including for our own earlier numbers: a band gap of ~1.5 pp is within
> what a training seed can move, and should not be called resolved from a single run.**
> Reproduce with `scripts/cdsd_multiseed.sh` → `scripts/cdsd_multiseed_contrasts.py`;
> per-seed results in `results/cdsd_multiseed_s{42,43,44}.csv`.

**What does not replicate:** `upper` vs `all`. CDSD resolves it in favour of `upper` (−1.47 pp, one third
of the trainable parameters); UASpeech does not resolve it in either direction. Reported as a
single-corpus result.

**Adapting the wrong band is worse than not adapting — on Qwen3-ASR.** On UASpeech, `lower` costs
+9.90 pp CER against the unadapted baseline and more than doubles mild-speaker CER (8.84 → 20.06), with
an interval excluding zero. Directionally consistent but unresolved on CDSD. On Whisper-small `lower` is
not harmful, merely the weakest band: it helps modestly (UASpeech CER −7.75, CI [−14.82, −0.60]).

**Two methodological results, and they generalise beyond this paper.**

*Probe-readout data efficiency is depth-dependent and architecture-specific.* Raising probe training data
5.4× (6,200 → 33,617 utterances) improves Whisper-small's **deepest** block by 9.15 pp — 1.87× its own
stack mean — while Qwen3-ASR's deepest gains 2.25 pp, 0.74× its mean. A low-volume probe therefore
manufactures an apparent late-stack difference between architectures (6.56 vs 1.85 pp of post-minimum
rise) that collapses to 0.25 pp at matched volume (2.07 vs 1.82). **Report the probe curve at more than
one training volume, or do not compare late-stack shape across architectures.**

*Whisper's padded frames are an information channel, not noise.* Whisper pads every input to 30 s and its
encoder ignores attention masks, so padded positions carry contextualised copies of the speech. In a
matched-pair experiment (same utterances, same taps, padding the only variable) the padded arm scored
**better** than the sliced arm — 53.54 vs 87.89 PER at the minimum — and both minimise at layer 10.

**We cannot reproduce the prior work's mid-stack plateau.** Probe configuration matches item by item,
unadapted baseline competence agrees to 1.5 pp under the decode convention both figures were produced
with (103.57 vs 105.08 CER; see the decode note below), and the discrepancy survives both padding
treatments and matched training volume. Reported as unexplained. Note the probe is a CTC readout over
encoder activations and never runs the autoregressive decoder, so **no probe result here depends on the
decode configuration.**

> **Decode note (2026-08-25).** Whisper-small's unadapted CDSD CER was first measured at 103.57 — above
> 100%, which invites reading it as hallucination. That figure was produced with greedy search and a flat
> 200-token budget on utterances averaging 2.4 s, leaving room for repetition loops whose every looped
> token scores as an insertion. Re-running **every** Whisper condition with repetition control
> (`no_repeat_ngram_size` 3, beam 5) and a duration-scaled cap gives the numbers above. The correction is
> strongly non-uniform — the unadapted base gains 26.01 pp, the adapted bands 1.85-4.63 pp — because an
> adapted model has little looping to suppress. **No placement conclusion changes on either corpus**;
> what changes is the size of every gap involving an unadapted or weakly adapted condition. Two verdicts
> moved and are reported at corrected values: CDSD `all` - `middle` (was resolved, now spans zero) and
> UASpeech CER `lower` - base (was spanning zero, now resolved). A third claim was withdrawn: WER and CER
> no longer rank `middle`/`upper` in opposite orders - that disagreement was an artefact. Both settings
> remain reproducible via `--decode {fixed,published}` in `scripts/eval_whisper_variants.py`; see
> `scripts/whisper_decode_check.py` and `results/whisper_decode_check.csv`.

**Two constraints reported rather than buried.** The encoder emits one frame per ~77 ms against
Whisper-small's 20 ms; against this corpus's forced alignments 49.4% of phones are shorter than a single
frame, so frame-level phoneme-boundary probing is *ill-posed* on this architecture, not merely weak. And
shallow adaptation damages acoustically atypical speakers via **language-prior takeover** — the adapted
model emits fluent, corpus-typical text unrelated to the audio while the unadapted model transcribes
correctly.

## Layout

```
scripts/            experiment pipeline (see "Pipeline" below for run order)
results/            aggregate result CSVs — metrics only, no audio and no transcripts
results/splits/     split manifests: utterance ids and split assignment, with fingerprints
results/regression/ the environment-check manifest and its expected scores
figures/            the paper's four figures, PDF + PNG, regenerable from results/ alone
```

Figures are built by `python scripts/make_figures.py` and read nothing but `results/` — no GPU, no
corpus access, no model weights.

### `results/`

| File | Contents |
|---|---|
| `qwen3asr_zh_probe_layerwise.csv` | Long-form probe metrics, all layers, all probes (CDSD) |
| `probeB_base_layerwise.csv`, `probeB_zh_aug_layerwise.csv` | Probe B — phoneme-identity CTC probe, tone-sensitive and tone-agnostic PER |
| `probeC_base_layerwise.csv`, `probeC_zh_aug_layerwise.csv` | Probe C — recognition-difficulty linear probe, macro-F1 |
| `qwen3asr_zh_condition_divergence.csv` | Cross-condition divergence by layer (distributional, per-speaker, strictly paired) |
| `qwen3asr_zh_target_sweep.csv` | CDSD four-band sweep — base, `upper`, `middle`, `all` |
| `qwen3asr_zh_target_sweep_lower.csv` | CDSD `lower` band (scored in a later run) |
| `qwen3asr_uaspeech_target_sweep.csv` | UASpeech five-variant sweep |
| `band_bootstrap_cis.csv` | Paired speaker-bootstrap CIs, both corpora, marginal and paired contrasts |
| `synth_cell_skew.csv` | Training-cell composition analysis behind the CDSD_08 failure mode |
| `seed_speaker_analysis.csv` | Two-dimensional speaker × seed bootstrap (UASpeech), separating seed variance from speaker-sampling variance |
| `seedsweep/{all,upper,middle,lower}_s{42,43,44}.csv` | UASpeech per-band, per-seed sweep results (12 runs) |
| `whisper_uaspeech_sweep_fixed_s{42,43,44}.csv` | Whisper-small × UASpeech, three seeds, corrected decode |
| `cdsd_multiseed_s{42,43,44}.csv`, `whisper_cdsd_multiseed_s{42,43,44}.csv` | CDSD four-band sweeps at three seeds, both architectures |
| `alpha_matched_s{42,43,44}.csv` | The α-matched placement × capacity 2×2 (§6.7) |
| `canonical_numbers.json` | Every number the manuscript quotes, regenerated by `scripts/canonical_numbers.py` |

**Whisper-small arm** (added 2026-08-17 → 08-20; this is what converted the central comparison from
cross-paper to internally controlled):

| File | Contents |
|---|---|
| `whisper_band_sweep.csv` | Whisper-small four-band sweep on CDSD, same manifest / split / gate / recipe as the Qwen3-ASR arm |
| `whisper_probe_layerwise.csv` | Whisper-small depth probe, 13 taps, 6,200 probe-training utterances |
| `whisper_volume_probe_layerwise.csv` | Same probe at **33,617** training utterances (matched to the prior work's scale) |
| `qwen_volume_probe_layerwise.csv` | Qwen3-ASR depth probe, 25 taps, at the same 33,617 — the matched-volume comparison |
| `padding_replicate_SLICED.csv`, `padding_replicate_PAD750.csv` | Matched-pair padding experiment — same utterances, same taps, padding the only variable |

### Reproducibility artefacts

| File | Contents |
|---|---|
| `decode_spec.json` | **The frozen decode specification.** Section 6.6.1 shows generation settings alone moving Whisper-small CER by tens of points, so the decode is part of the result. Pins model and checkpoint identifiers, library and CUDA versions, prompt language and task tokens, beam width, `no_repeat_ngram_size`, the duration-scaled output-length rule, stopping criteria, sampling flags, text normalisation, CER/WER tokenisation, the checkpoint key-remapping shim, and the CPU-merge requirement |
| `regression/regression_manifest.csv` | 180 CDSD test utterances, 30 per test speaker, selected deterministically |
| `regression/regression_expected.json` | Expected CER for that manifest under four cells — Qwen3-ASR base and `upper`, Whisper-small base and `all`. **The single most useful file here:** a mismatch means the environment differs from the one the paper's numbers came from, most likely a checkpoint that loaded with unmatched keys |
| `splits/*_splits.csv` | Utterance id, speaker and split assignment for all six manifests. No paths, no transcripts, no durations |
| `splits/MANIFEST_FINGERPRINTS.json` | SHA-256 of each source manifest and of the released rows, plus per-split utterance and speaker counts and a verified speaker-disjointness flag |
| `cdsd_interaction_speaker_tallies.csv` | Per-speaker character edit distance and reference length for each (model, band, seed) of the interaction test. **This is what makes the paper's primary claim reproducible** without releasing predictions — see below |

**On the interaction test.** `scripts/interaction_test.py` computes the paper's primary claim, and it
originally read per-utterance prediction files, which are corpus content this repository does not
release. The test never uses an individual utterance: micro and macro CER are exact functions of two
integers per speaker, the total edit distance and the total reference length. Those tallies ship, so the
released path reproduces the published numbers **exactly** rather than approximately — DoD −4.707 under
micro CER, −4.604 under macro, exact sign-flip p = 0.031. Pass `--from-preds` to recompute from the
prediction files instead, which needs the working tree and the corpus.

**On the two 2.1 MB files.** `splits/cdsd_lora_manifest_splits.csv` and
`splits/cdsd_probe_manifest_FULL_splits.csv` are the largest files here. Both are plain text: 47,687
rows of `uid,speaker_id,split`. A file-size scan will flag them; they contain no corpus data.

Per-utterance prediction files are **deliberately not included** — see Data and licensing.

---

## Pipeline

Run order. Paths assume a CANDOR-style tree; every script takes explicit `--` arguments.

**1. Encoder geometry** (a precondition, and a finding — establishes the ~77 ms frame period)
```bash
python scripts/probe_encoder_geometry.py
```

**2. Probe dataset** — forced alignments → per-utterance phone sequences, with a `.lab` verification gate
```bash
python scripts/build_probe_dataset.py --subset 1h
```
`--subset` is **not** optional in practice: see the CDSD subset collision under Gotchas.

**3. Layer features** — dump per-layer encoder activations
```bash
python scripts/extract_layer_features.py --layers 0-24
```

**4. Probes**
```bash
python scripts/train_probe_b_ctc.py          # phoneme identity (PER by layer)
python scripts/decode_for_difficulty.py      # difficulty labels for Probe C
python scripts/train_probe_c_difficulty.py   # recognition difficulty (macro-F1 by layer)
```

**5. Cross-condition divergence**
```bash
python scripts/build_condition_manifests.py
python scripts/condition_similarity.py
```

**6. Adaptation sweep** — the four bands, matched parameter budget
```bash
python scripts/train_lora_target_sweep.py --variant upper  --corpus cdsd
python scripts/train_lora_target_sweep.py --variant middle --corpus cdsd
python scripts/train_lora_target_sweep.py --variant all    --corpus cdsd
python scripts/train_lora_target_sweep.py --variant lower  --corpus cdsd
```
For the second corpus, build the split first, then sweep with `--corpus uaspeech`:
```bash
python scripts/build_uaspeech_dataset.py
python scripts/train_lora_target_sweep.py --variant upper --corpus uaspeech \
    --manifest results/probe/uaspeech_lora_manifest.jsonl
```

**7. Merge and score**
```bash
python scripts/merge_adapter_export.py --adapter <run>/adapter --out <run>/merged
python scripts/eval_lora_variants.py \
    --variant base=Qwen/Qwen3-ASR-1.7B --variant upper=<run>/merged ... \
    --manifest <corpus manifest> --split test
```

**8. Statistics**
```bash
python scripts/bootstrap_band_cis.py --b 10000
python scripts/seed_speaker_analysis.py        # two-dimensional speaker x seed bootstrap
```

---

## The Whisper-small arm (steps 9–12)

Steps 1–8 characterise Qwen3-ASR. These steps repeat the prior work's architecture under the *same*
pipeline, which is what makes the cross-architecture comparison internally controlled rather than
cross-paper. Run them after step 8.

**9. Whisper band sweep** — identical recipe, bands defined proportionally over 12 blocks
```bash
python scripts/train_lora_whisper_sweep.py --variant all    --corpus cdsd
python scripts/train_lora_whisper_sweep.py --variant middle --corpus cdsd   # blocks 5-8
python scripts/train_lora_whisper_sweep.py --variant upper  --corpus cdsd   # blocks 9-12
python scripts/train_lora_whisper_sweep.py --variant lower  --corpus cdsd   # blocks 1-4
python scripts/eval_whisper_variants.py --variant base=openai/whisper-small ...
```
The trainer imports its audited gates from `train_lora_target_sweep.py`, so both arms share one
implementation of the disk, smoke, epoch and completion checks.

**10. Whisper depth probe** — 13 taps, sliced to real frames
```bash
python scripts/extract_layer_features_whisper.py --tag whisper_base
python scripts/train_probe_b_ctc.py --tag whisper_base --out results/whisper_probe_layerwise.csv
```
The extractor slices each utterance to `ceil(duration / 0.02)` frames **before storing anything**.
Whisper pads to a fixed 1500 frames regardless of real duration; storing that raw would both explode the
feature set and train the probe on ~92% padding.

**11. Padding control** — one extraction pass, two shard sets, padding the only variable
```bash
python scripts/padding_replicate_extract.py     # writes whisper_pad1500 + whisper_slice_matched
```

**12. Matched-volume probe** — both architectures at the prior work's data scale
```bash
bash scripts/run_volume_perlayer.sh          # Whisper, 13 taps, one process per layer
bash scripts/run_qwen_volume_perlayer.sh     # Qwen3-ASR, 25 taps, one process per 5-layer group
python scripts/compare_volume_curves.py      # shape summary + verdict
```
**Both runners deliberately use one process per layer group.** On unified-memory hardware a single
process sweeping all layers is OOM-killed at the first layer transition, because one layer's allocation
does not fully release before the next load begins. `train_probe_b_ctc.py` additionally reserves a CUDA
slab (`PROBE_CUDA_RESERVE_GB`, default 6) before the host-side feature load, since a large host
allocation otherwise leaves CUDA unable to allocate at model-creation time.

---

## Environment

**To reproduce the released artefacts, you need two packages and no GPU.**

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-analysis.txt

python scripts/make_figures.py        # the paper's four figures
python scripts/canonical_numbers.py   # results/canonical_numbers.json
python scripts/interaction_test.py    # the paper's primary claim
python scripts/lora_param_count.py    # the Section 6.7 parameter budgets (standard library only)
```

Everything above reads `results/` alone: no corpus, no model weights, no CUDA.

**To run the full pipeline** you additionally need the corpora, which are not redistributable, and a
GPU environment: `requirements-pipeline.txt`. Note that training and evaluation require **separate**
environments — the trainer needs a `transformers` release the packaged inference path rejects, and the
evaluator needs one that loads the Qwen3-ASR checkpoint without silently initialising it at random.
`results/decode_spec.json` pins the exact versions, and that file is the authority.


**Two virtual environments are required, and they are not interchangeable.**

| Purpose | transformers | Why |
|---|---|---|
| Probing, feature extraction, LoRA training, adapter merge | **5.14.1** | First version shipping `qwen3_asr`. Absent in 5.0.0 |
| Decoding and scoring | **4.57.6** (with `qwen_asr`) | The only decode path validated here — see below |

Decoding goes through Qwen's own `qwen_asr.Qwen3ASRModel.transcribe`. Measured on 369 utterances
overlapping recorded fair-eval predictions:

| Decode path | CER |
|---|---|
| recorded reference decode | 41.87% |
| `transformers` `apply_chat_template` + `generate` | **124.57%** — do not use |
| `qwen_asr.Qwen3ASRModel.transcribe` | **32.38%** |

`merge_adapter_export.py` therefore writes the merged checkpoint back in the original `thinker.*`
namespace so `qwen_asr` can load it.

---

## Gotchas

These cost real time. They are documented here so they cost no one else any.

**1. Loading a Qwen3-ASR checkpoint under transformers 5.x silently produces a RANDOM model.**
The base repo and every CANDOR fine-tune store weights under the pre-restructure `thinker.*` namespace.
transformers 5.14's `Qwen3ASRForConditionalGeneration` expects `model.*` and ships **no**
`_checkpoint_conversion_mapping`, so `from_pretrained` yields **exact key overlap 0/710** — a fully
randomly-initialised model, signalled only by a warning block that is easy to scroll past.
`scripts/qwen3asr_compat.py::load_qwen3_asr` performs the remap and **asserts zero unmatched parameters**,
refusing to load otherwise. Always load through it.

Two further traps in the same area: `AutoConfig.from_pretrained` returns **default** sub-configs (the real
ones are nested under `thinker_config`) whose values coincide with the checkpoint on most fields but set
`audio_config.output_dim` to 3584 against the checkpoint's 2048, building the projector at the wrong
width; and the sub-config `model_type` strings are stale (`qwen3_asr_audio_encoder` → `qwen3_asr_encoder`,
`qwen3_asr_text` → `qwen3`).

**2. CDSD ships two subset directories whose utterance IDs collide.**
The `1h` subset holds 55,851 utterances from 44 speakers (35.06 h); `10h` holds 92,730 from 8 speakers
(64.32 h). **6,103 `(speaker, utt_id)` pairs name one file in EACH subset with different transcripts.**
Identifiers are unique within either subset; the collision exists only across them. Any alignment or
feature store keyed on `(speaker, utt_id)` alone will silently mix them. Key on
`(speaker, subset, utt_id)`. This study uses the `1h` subset throughout — which is also the subset the
prior work used, identifiable from its reported ~55,850 utterances.

**3. Scoring conventions differ by language and are not interchangeable.**
`eval_lora_variants.py` defaults to the Mandarin convention: no case folding, character tokens, and WER
== CER because Mandarin has no orthographic word boundaries. For alphabetic languages pass
**`--case-fold --tokenise word`**. Without case folding, a model emitting `Command` against a reference
`command` scores every initial letter as a substitution; without word tokens, "WER" is just CER again.

**4. Training loss does not predict held-out band ordering.**
On UASpeech the run-mean training losses ordered exactly as held-out CER did. On CDSD they did not —
`all` reached a lower training loss than `upper` and scored worse on held-out speakers. Compare bands on
held-out CER only.

**5. Marginal confidence intervals are useless at this sample size.**
With 5–6 test speakers, marginal per-variant bootstrap CIs span 33–54 pp. Only **paired** contrasts, in
which speaker difficulty cancels within a replicate, are interpretable. `bootstrap_band_cis.py` uses one
shared set of speaker resamples per corpus so every variant and pair is scored on identical replicates.

**6. UASpeech splits are speaker-disjoint only.**
Every speaker reads the same 449-word list, so a text-disjoint split is impossible by construction. This
is a closed-vocabulary task by design and the numbers should be read as such.

**7. On unified-memory hardware, sweep one layer group per PROCESS.**
On a GB10 the host and the accelerator draw on one memory pool. A single process sweeping all layers is
OOM-killed at the *first* layer transition, because one layer's allocation does not fully release before
the next load begins — and a SIGKILL writes nothing to the log, so the failure looks identical to a slow
load. Both volume runners therefore spawn one process per layer group and skip groups whose output CSV
already exists, which also makes them resumable. `train_probe_b_ctc.py` separately reserves a CUDA slab
(`PROBE_CUDA_RESERVE_GB`, default 6) *before* the host-side feature load: without it, a ~66 GB host
allocation leaves CUDA unable to allocate at model-creation time even while reporting free memory.

**8. Whisper's encoder ignores attention masks, and its padding is not inert.**
Whisper pads every input to 30 s / 1500 frames and attends across the whole window, so padded positions
carry contextualised copies of the speech. Two consequences. Storing raw 1500-frame activations trains
the probe on ~92% padding, which fails *silently* — the resulting curve describes the model's treatment
of silence. But slicing is not free either: in the matched-pair control the padded arm scored **better**
than the sliced arm (53.54 vs 87.89 PER at the minimum). Whichever you choose, choose it deliberately and
report it, because it moves the absolute numbers by tens of points while leaving the knee where it is.

**9. Whisper's `pad_token_id` aliases its EOT (both 50257).**
Masking supervision by token *value* therefore deletes every genuine terminal EOT while the Qwen3-ASR arm
trains its EOS — an accidental confound on the objective itself. Mask with the attention mask, not by
value. Relatedly, the seq2seq collate must not keep the start-of-transcript token, since
`shift_tokens_right` inserts its own; leaving it in shifts the optimisation target one position and cost
a smoke loss of 5.70 against 1.71 once fixed.

**10. A `--seed` flag that is read after the model is built does nothing.**
In an earlier revision the seed reached `TrainingArguments` but not `get_peft_model`, so every "different
seed" shared identical LoRA initialisation and the seed study measured data-order variance alone. Set the
seed *before* model construction, and verify by confirming two seeds produce different initial weights.

---

## Data and licensing

**No audio, transcripts, alignments or per-utterance predictions are included in this repository.**
Only aggregate metrics. This is a licensing requirement, not a convenience.

| Corpus | Terms |
|---|---|
| **CDSD** (Chinese Dysarthria Speech Database) | Custom academic licence with **explicit no-redistribution**. Any publication using this audio must credit **© Su-Jing Wang**. |
| **UASpeech** | Licensed from the University of Illinois; not redistributable here. |

Obtain both from their custodians directly. The scripts expect locally prepared manifests and will not
download data.

Model weights (base and adapted) are not included.

---

## Citation

The paper is not yet published; a citation block and DOI will be added here on acceptance. Until then,
cite the repository and note the corpus credit above.

---

## Contact

Bernard Muller — Technical PI, CANDOR — `bernard@scottmorganfoundation.org`

Issues and questions: use the repository issue tracker.
