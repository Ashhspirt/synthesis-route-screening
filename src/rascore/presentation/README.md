# LecTrace presentation

The lecture follows three saved validation molecules through molecular parsing, feature encoding, neural activations and ensemble decisions. It contains four selected computational states, 63 navigation steps and eight scientific illustrations.

From the repository root, run `python scripts/serve_demo.py`. The prebuilt public viewer requires only the Python standard library. Use arrow keys to move and E to open the variable panel.

To rebuild using the installed ML environment:

```sh
python -m pip install src/rascore/presentation/vendor/lectrace-1.1.30-py3-none-any.whl
python src/rascore/presentation/build_presentation.py
python scripts/sync_demo.py
```

The build reruns frozen inference on three validation examples and checks the saved probabilities. It does not retrain or rescore the official test. `presentation_checks.json` records the checks; `figures/figure_sources.json` records figure sources.
