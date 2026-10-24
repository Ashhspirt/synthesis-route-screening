# Sources and third-party notices

The training workflow is an independent PyTorch implementation informed by RAscore. Count fingerprints are an established representation. AI assisted planning, implementation and writing.

| Component | Role and source | Attribution |
| --- | --- | --- |
| RAscore | Historical synthesis-planner labels and task definition; [pinned source](https://github.com/reymond-group/RAscore/tree/cb77db503ee5cbf0e8bb8963df6e5b76b3a94f06) | Thakkar et al., Chemical Science, 2021. The upstream MIT notice is preserved in `src/rascore/licenses/RAscore_LICENSE.txt`. |
| ChEMBL | Public molecular structures represented in the RAscore data | Original ChEMBL data attribution and applicable terms remain in force. The RAscore software license does not replace the terms of third-party data. |
| AiZynthFinder | Planner used upstream to generate historical route-found labels | Genheden et al., Journal of Cheminformatics, 2020. This repository predicts historical labels; it does not run a retrosynthesis search. |
| RDKit | Molecule parsing, scaffolds, Morgan fingerprints, descriptors and drawings | [RDKit](https://www.rdkit.org/). |
| PyTorch, NumPy, pandas, scikit-learn, Matplotlib, psutil and joblib | Training, metrics, figures and resource measurements | Installed dependencies; recorded versions are in `src/rascore/requirements-lock.txt`. Their individual licenses apply. |
| LecTrace 1.1.30 | Execution-trace presentation and browser viewer | PraiseGod D. Adesanmi; [official source](https://github.com/praisegee/lectrace), MIT. The official wheel and license are retained under `src/rascore/presentation/vendor/`. |

Published molecular rows, saved predictions and weights derive from the public RAscore study data. The private building-block inventory used upstream is not included. No new blanket license is imposed on third-party software or data. Academic report files, cover pages and personal reflections are omitted from this code release.
