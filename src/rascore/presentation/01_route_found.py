"""MD6329 synthesis planning for early drug discovery: four selected computation states."""
from pathlib import Path
import numpy as np
import torch
from lectrace import text, image
from _model_demo import load_demo, molecule_state, features, layer_summary, decision, saved_results
HERE = Path(__file__).resolve().parent
FIGURES = HERE / 'figures'


def main():
    demo = load_demo()  # @invisible
    text(
        '# Can a small neural network help prioritize synthesis route searches in early drug discovery?'
    )
    text(
        '**Molecular synthesis route screening for early drug discovery**  \nCPU-based deep learning study'
    )
    text(
        '**Healthcare problem.** New medicines start as candidate molecules. Before a promising candidate can enter biological testing, and eventually preclinical and clinical development for patients, chemists must find a practical way to make it. A single difficult reaction can stall that path for weeks.'
    )
    text(
        '**Screening problem.** Computer-aided retrosynthesis planners can search for routes, but running a multistep planner for every candidate costs time. A fast score can decide which candidates to search first, but missing a useful drug candidate has a cost. Label **1** means the historical AiZynthFinder run found a route; label **0** means it did not under its search conditions.'
    )
    text(
        '**Research questions.** Does a lightweight MLP improve on simple classifiers? Do substructure counts and whole-molecule descriptors help on unfamiliar scaffolds? What does reducing missed candidates cost in additional searches?'
    )
    image(str(FIGURES / 'pipeline.png'), width='100%')
    text(
        '**Study design.** Use 20,000 ChEMBL-derived, bioactivity-related development molecules, repeated random and scaffold splits, and three training seeds. Validation selects the configuration and threshold. A separate 19,935-molecule test evaluates the frozen comparison. Four selected states follow the same three validation molecules from structure to decision.'
    )

    text(
        '## State 1 — Turn molecular structures into labeled inputs'
    )
    molecules = molecule_state(demo)  # @inspect molecules @stepover
    image(str(HERE / 'demo_molecules.png'), width='100%')
    text(
        '**What changes?** Stored SMILES become molecular structures paired with historical route-found labels. These are the first three saved validation rows, chosen without looking for successful predictions. Expand `molecules` to inspect their source IDs, labels and structure strings.'
    )
    text(
        '**Why it matters.** A scaffold split keeps shared core structures together, testing transfer beyond those cores. A route-found label describes a planning outcome; reaction conditions and yields still require separate investigation.'
    )

    text(
        '## State 2 — Encode local patterns and whole-molecule properties'
    )
    x, feature_state = features(demo)  # @inspect feature_state @clear molecules @stepover
    text(
        '**Core operations inside `features`.**\n\n```python\nscaled = (descriptors - mean) / scale\nlocal = np.log1p(counts)\nx = np.concatenate([local, scaled], axis=1)\n```\n\nCounts preserve repeated environments. Five descriptors use training-only scaling. The matrix has **3 rows × 1,029 columns**: 1,024 fingerprint positions plus molecular weight, rings, rotatable bonds, donors and acceptors.'
    )
    image(str(FIGURES / 'feature_encoding.png'), width='100%')
    text(
        '**Read the actual example.** One hashed position occurs ten times in the first molecule. Binary encoding gives 1; counts retain 10 before log(1 + count) gives about 2.398. This is a hashed bin, not a unique chemical group. The descriptor panel shows the same molecule before and after training-set scaling.'
    )
    image(str(FIGURES / 'representation_comparison.png'), width='100%')
    text(
        '**Why choose counts?** Fixed fingerprints keep repeated CPU experiments affordable. Counts improve development AUROC in all paired splits, while stronger regularization is not consistently better. Error bars show variation across three split means. A graph model could learn its encoding, but needs a separate comparison.'
    )

    text(
        '## State 3 — Transform features into neural predictions'
    )
    text(
        '**What happens.** We now run the frozen seed-42 network one layer at a time on the 3 × 1,029 input matrix. Each hidden block is a linear layer followed by ReLU; dropout is inactive in evaluation mode and no gradients are computed. Watch `layer1`, `layer2` and `layer3` appear in the variable panel.'
    )
    net = demo['models'][0]  # @stepover
    inputs = torch.from_numpy(x)  # @clear feature_state
    with torch.no_grad():
        h1 = net[1](net[0](inputs))  # @stepover
        layer1 = layer_summary(h1)  # @inspect layer1 @stepover
        h2 = net[4](net[3](h1))  # @stepover
        layer2 = layer_summary(h2)  # @inspect layer2 @stepover
        h3 = net[7](net[6](h2))  # @stepover
        layer3 = layer_summary(h3)  # @inspect layer3 @stepover
        logit = net[9](h3)  # @stepover
        seed42_logit = logit.squeeze(1).tolist()  # @inspect seed42_logit
        seed42_probability = torch.sigmoid(logit).squeeze(1).tolist()  # @inspect seed42_probability
    text(
        '**What changes.** The representation shrinks from **1,029 → 256 → 128 → 64 → 1** values per molecule. ReLU sets negative pre-activations to zero, so `active_units_per_molecule` counts how many hidden units fire for each molecule. The final logit is unbounded; sigmoid maps it into a probability between 0 and 1.'
    )
    image(str(FIGURES / 'network_activations.png'), width='100%')
    text(
        '**Repeat for all three seeds.** The same input passes through the seed-43 and seed-44 networks, giving one probability per molecule per model.'
    )
    with torch.no_grad():
        probabilities = np.stack([torch.sigmoid(model(inputs)).squeeze(1).numpy().astype(np.float64) for model in demo['models']], axis=1)  # @stepover
    assert np.allclose(probabilities[:, 0], seed42_probability, atol=1e-6)  # @invisible
    three_seed_probabilities = probabilities.round(4).tolist()  # @inspect three_seed_probabilities @clear layer1 layer2 layer3
    text(
        '**Why it matters.** Nonlinear layers combine local substructure patterns with global properties such as size and ring count. The smaller network saves about 77.5% of parameters but loses development AUROC. Final models restore epoch 2, where validation loss is lowest, despite later falls in training loss. The heatmap shows sampled units, not a chemical attribution.'
    )

    text(
        '## State 4 — Average predictions and make a screening decision'
    )
    text(
        '**What happens.** The three seed probabilities are averaged, then compared with a threshold frozen on validation data before testing.'
    )
    threshold = demo['threshold']  # @inspect threshold
    mean_probability = probabilities.mean(axis=1)
    ensemble_mean = mean_probability.round(4).tolist()  # @inspect ensemble_mean @clear seed42_logit seed42_probability
    positive = mean_probability >= threshold
    predicted_label = positive.astype(int).tolist()  # @inspect predicted_label
    historical_label = [m['historical_label'] for m in molecules]  # @inspect historical_label
    decisions = decision(demo, probabilities)  # @inspect decisions @stepover
    text(
        '**What changes.** Three probabilities per molecule collapse into one ensemble score, and the score becomes a yes/no action: **1** = search this candidate first, **0** = defer it. Compare `predicted_label` with `historical_label`. The `decision` call also checks that these means match the saved validation predictions.'
    )
    image(str(FIGURES / 'ensemble_decisions.png'), width='100%')
    text(
        '**Why it matters.** Averaging uses all three fits rather than trusting one initialization. The threshold, about **0.5844**, retains at least 90% of validation positives, encoding the preference for fewer missed drug candidates; equality counts as positive.'
    )
    image(str(FIGURES / 'threshold_tradeoff.png'), width='100%')
    text(
        '**Lecture 2, slides 92 and 99.** AUROC measures ranking; the operating threshold determines decisions. For the same count model, the recall policy avoids **1,323 misses** but adds **573 false positives** compared with the balanced policy. Threshold changes leave AUROC unchanged.'
    )

    text(
        '## What did the held-out comparison establish?'
    )
    results = saved_results()  # @invisible
    text(
        '**A small ranking improvement.** On 19,935 held-out molecules, count-ensemble AUROC is **0.8888**, versus **0.8852** for binary. The paired gain is **0.00362**, with 95% bootstrap interval **0.00232–0.00495**. Default count recall is **90.24%**. At fixed recall-oriented thresholds, counts avoid **160 misses** at the cost of **45 extra false positives**. These are saved evaluation results.'
    )
    image(str(FIGURES / 'structural_errors.png'), width='100%')
    text(
        '**Where it still struggles.** Recall is **88.82%** on scaffolds absent from development. Molecules with more than four rings have **26.51%** false negatives. Preserve review opportunities for deferred molecules, especially difficult structures.'
    )
    text(
        '**Healthcare interpretation and next step.** The model provides a low-cost priority score for later synthesis searches. Earlier synthesis triage could reduce avoidable delays before biological testing and help promising drug candidates progress toward new treatments; actual workflow and patient-access benefits need prospective validation. It does not produce a reaction route. Next, inspect planner traces for difficult structures and evaluate changes on newly reserved or external data. Chirality-aware fingerprints and graph representations are potential comparisons.'
    )
    text(
        '**AI assistance and sources.** AI assisted project planning, implementation and writing. The screening policy prioritizes fewer missed candidates. Sources: RAscore (Thakkar et al., 2021), AiZynthFinder (Genheden et al., 2020), RDKit and MD6329 Lecture 2. Data and software attribution are available in THIRD_PARTY_NOTICES.md.'
    )


if __name__ == '__main__':
    main()
