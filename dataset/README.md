# Datasets

This folder holds the five benchmarks used in BehaVERT. The data itself is on the
Hugging Face Hub ([`ShinSeungJ/behavert-data`](https://huggingface.co/datasets/ShinSeungJ/behavert-data))
and is downloaded here with the `behavert` command, run from the repository root:

```bash
behavert download                    # all five benchmarks, about 12 GB
behavert download calms21 mabe22     # or only the ones you need
```

Every command (`train`, `evaluate`, `decode --config`, `attention`, `embed`) reads
this folder by default. To keep the data elsewhere, pass `--dir` to `download`
and `--data-root` to the other commands, or set `BEHAVERT_DATA`.

## Contents

Before downloading, each benchmark folder holds only `sample.mp4`, a short video
from that dataset. The download adds the data and a README that describes the files
and how they were derived from the original release.

| Folder | Benchmark | Animals | Keypoints | Tasks | Size |
|---|---|---|---|---|---|
| `calms21/` | CalMS21 | 2 mice, 2D | 7 per mouse | Task 1, Task 3, self-supervised pretraining | 4.8 GB |
| `mabe22/` | MABe22 mouse triplets | 3 mice, 2D | 12 per mouse | strain, lights, experiment day, time of day, 9 behaviors | 3.6 GB |
| `pair24m/` | PAIR-R24M | 1–2 rats, 3D | 12 per rat | single-animal behavior, social interaction | 2.1 GB |
| `deepethogram/` | DeepEthogram | 1 mouse, 2D | 11 (our YOLOv12 predictions) | 6 behaviors | 1.4 GB |
| `sbea/` | SBeA Shank3B | 2 mice, 3D | 16 per mouse | genotype pair, behavior (our annotations) | 0.3 GB |

The configs in [`../configs`](../configs) refer to these folders, e.g.
`configs/mabe22/state.yaml` reads `mabe22/mabe22_kpts.pt` and
`mabe22/task/strain/mabe22_strain.pt`.

## License

- **Our own work** (the DeepEthogram keypoints and the SBeA behavior annotations)
  is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **CalMS21 and PAIR-R24M files** are adapted from releases licensed under CC BY 4.0
  and are shared under the same license.
- **MABe22 files, the DeepEthogram labels and videos, and the SBeA coordinates**
  are derived from their original releases and remain subject to those releases'
  terms. They are shared for research use, with credit to the original authors.

The sample videos come from the same releases and fall under the same terms.
Details are in [`LICENSE`](LICENSE) and in each benchmark's README.

## Citation

Please cite the original dataset of every benchmark you use, together with BehaVERT:

| Benchmark | Original source |
|---|---|
| CalMS21 | Sun et al., *The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions*, NeurIPS 2021 Datasets & Benchmarks |
| MABe22 | Sun et al., *MABe22: A Multi-Species Multi-Task Benchmark for Learned Representations of Behavior*, ICML 2023 |
| PAIR-R24M | Marshall et al., *The PAIR-R24M Dataset for Multi-animal 3D Pose Estimation*, NeurIPS 2021 Datasets & Benchmarks |
| DeepEthogram | Bohnslav et al., *DeepEthogram, a machine learning pipeline for supervised behavior classification from raw pixels*, eLife 2021 |
| SBeA | Han et al., *Multi-animal 3D social pose estimation, identification and behaviour embedding with a few-shot learning framework*, Nature Machine Intelligence 2024 |

```bibtex
@article{shin2026behavert,
  title   = {BehaVERT: A Transformer-Based Motion Language Model for Decoding Behavioral Semantics in Mice},
  author  = {Shin, Seung Jae and Lee, Junesu and Ko, Chang and Kim, Daesoo},
  journal = {International Journal of Computer Vision},
  volume  = {134},
  pages   = {303},
  year    = {2026},
  doi     = {10.1007/s11263-026-02834-y}
}
```
