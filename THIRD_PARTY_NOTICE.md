# Third-party notices and scholarly attribution

This document records the identifiable external software and research ideas
used by PSPA-DT. It is provided for attribution and release review; it is not
legal advice. A citation does not replace compliance with a software license.

## Historical decoder provenance

An earlier internal research version used a 3-D adaptation of EMCAD. That
implementation and its `MSDC3D`, `MSCB3D`, `EUCB3D`, `LGAG3D`, `CAB3D`,
`SAB3D`, and `EMCAD*` classes are **not distributed in this repository**.
They were removed because the official EMCAD license restricts redistribution.
For scholarly attribution, the official source is:

- Md Mostafijur Rahman, Mustafa Munir, and Radu Marculescu, **“EMCAD:
  Efficient Multi-scale Convolutional Attention Decoding for Medical Image
  Segmentation,”** CVPR 2024, pp. 11769–11779.
- Paper: <https://openaccess.thecvf.com/content/CVPR2024/html/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.html>
- Official code: <https://github.com/SLDGroup/EMCAD>
- Official license: <https://github.com/SLDGroup/EMCAD/blob/main/LICENSE>

The official EMCAD code is governed by the **UT Austin Research License**, not
by Apache-2.0 or MIT. As published on 7 September 2026, that license permits
academic, research, experimental, and personal use, excludes commercial use,
and states that licensees may not distribute or transfer copies of the
software. It also places derivative products under the same restrictions.

The current `src/pspa_dt/network/Decoder.py` is a new project-owned
implementation named `PSPADecoder3D`. It uses standard PyTorch primitives and
was written independently for the public release. It does not contain the
earlier EMCAD-derived source. Because replacing the decoder changes the model,
results from the public PSPA decoder must not be represented as numerically
identical to results produced by the earlier internal decoder.

For an academic repository that does not have redistribution permission,
publish only a reference to the official EMCAD source instead of committing
the adapted `Decoder.py`. Readers should obtain any permitted upstream code
directly from the official repository and comply with its license. Merely
identifying the official source does not create a right to redistribute it.

The repository-level `LICENSE` file does not override the EMCAD license for
code derived from EMCAD.

## nnU-Net v2

PSPA-DT uses nnU-Net for experiment planning, preprocessing infrastructure,
training orchestration, sliding-window inference, and medical-image I/O.
nnU-Net source code is not vendored; `nnunetv2==2.6.2` is installed as a
runtime dependency.

- Project: <https://github.com/MIC-DKFZ/nnUNet>
- License: Apache License 2.0
- License text: <https://github.com/MIC-DKFZ/nnUNet/blob/master/LICENSE>
- Fabian Isensee, Paul F. Jaeger, Simon A. A. Kohl, Jens Petersen, and
  Klaus H. Maier-Hein, **“nnU-Net: a self-configuring method for deep
  learning-based biomedical image segmentation,”** Nature Methods 18,
  203–211 (2021). <https://doi.org/10.1038/s41592-020-01008-z>

The code under `src/pspa_dt/training`,
`src/pspa_dt/preprocessing`, and `src/pspa_dt/inference` extends
nnU-Net through subclassing and public APIs.

## Mamba and selective state spaces

`src/pspa_dt/network/model/SS2D_Encoder.py` imports the `Mamba` class
from the external `mamba-ssm` package. The package is not vendored.

- Project: <https://github.com/state-spaces/mamba>
- License: Apache License 2.0
- License text: <https://github.com/state-spaces/mamba/blob/main/LICENSE>
- Albert Gu and Tri Dao, **“Mamba: Linear-Time Sequence Modeling with
  Selective State Spaces,”** arXiv:2312.00752 (2023).
  <https://arxiv.org/abs/2312.00752>

## VMamba / visual selective scan

The multidirectional selective-scan design represented by `SS2D_encoder`,
`SS3D_encoder`, and `VSSM3D` is adapted from or inspired by the Visual State
Space / SS2D formulation popularized by VMamba, with local extensions for 3-D
volumes.

- Project: <https://github.com/MzeroMiko/VMamba>
- License: MIT License
- License text: <https://github.com/MzeroMiko/VMamba/blob/main/LICENSE>
- Yue Liu, Yunjie Tian, Yuzhong Zhao, Hongtian Yu, Lingxi Xie, Yaowei Wang,
  Qixiang Ye, and Yunfan Liu, **“VMamba: Visual State Space Model,”**
  Advances in Neural Information Processing Systems 37 (NeurIPS 2024).
  <https://arxiv.org/abs/2401.10166>

If any lines in `SS2D_Encoder.py`, `VSSM.py`, or `archi_utils.py` were copied
or adapted from VMamba rather than independently implemented, the VMamba
copyright notice and MIT permission notice must be retained with those files.

## clDice

The connectivity loss in
`src/pspa_dt/training/trainers.py` implements differentiable 3-D soft
skeletonization and centerline Dice based on:

- Suprosanna Shit, Johannes C. Paetzold, Anjany Sekuboyina, Ivan Ezhov,
  Alexander Unger, Andrey Zhylka, Josien P. W. Pluim, Ulrich Bauer, and
  Bjoern H. Menze, **“clDice—a Novel Topology-Preserving Loss Function for
  Tubular Structure Segmentation,”** CVPR 2021, pp. 16560–16569.
  <https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html>
- Reference implementation: <https://github.com/jocpae/clDice>
- Reference implementation license: MIT License

The PSPA-DT implementation includes project-specific handling for multiclass
targets, ignore labels, deep supervision, and activation checkpointing.

## Haar wavelet transform

`src/pspa_dt/network/wavelet.py` implements a one-level separable
3-D Haar discrete wavelet transform and its inverse directly with PyTorch
tensor operations. No third-party wavelet library is bundled. The Haar
transform is a standard mathematical construction; the surrounding wavelet
fusion, gating, and residual integration are part of the PSPA-DT research code.

## Components currently recorded as project code

The following components have no identifiable external source recorded in the
available source files or Git history and are therefore documented here as
project-specific implementations pending confirmation by the authors:

- `PSPADTBackbone`, `PSPADTDynamicBackbone`, `PSPADTJointFlux`, `PSPADTNoDeepSupervision`, and `PSPADTRefineBackbone` in
  `network/UnetMix.py`;
- `LoENm3D`, `LoE3D`, and associated channel-attention blocks in
  `network/model/LoE.py`;
- `MultiFrequencyChannelAttention3D` in `network/MFCA.py`;
- `WaveletSSM3D`, `DeltaWaveletSSM3D`, and the PSPA-DT wavelet integration;
- flux normalization, regression adaptation, export, and evaluation code.

Before release, the authors must confirm that these files were written by the
project authors and were not copied or adapted from an unrecorded repository.
If an external source is identified, add its copyright, license, repository,
and paper citation here before redistribution.

## Runtime dependencies

These packages are installed as dependencies and are not vendored here:

| Dependency | Role | Upstream license | Upstream project |
|---|---|---|---|
| PyTorch | tensor operations and neural networks | BSD-style | <https://github.com/pytorch/pytorch> |
| mamba-ssm | selective-scan kernels and Mamba module | Apache-2.0 | <https://github.com/state-spaces/mamba> |
| timm | initialization/model utilities | Apache-2.0 | <https://github.com/huggingface/pytorch-image-models> |
| einops | tensor rearrangement | MIT | <https://github.com/arogozhnikov/einops> |
| SciPy | statistics and numerical routines | BSD-3-Clause | <https://github.com/scipy/scipy> |
| SimpleITK | medical-image I/O and geometry | Apache-2.0 | <https://github.com/SimpleITK/SimpleITK> |
| nnU-Net | medical segmentation framework | Apache-2.0 | <https://github.com/MIC-DKFZ/nnUNet> |

Each dependency remains governed by its own license and may include additional
transitive dependencies and notices. Binary CUDA distributions can have
additional terms; consult the package and CUDA distribution documentation.

## Citation records

```bibtex
@article{isensee2021nnunet,
  title   = {nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author  = {Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and Petersen, Jens and Maier-Hein, Klaus H.},
  journal = {Nature Methods},
  volume  = {18},
  pages   = {203--211},
  year    = {2021},
  doi     = {10.1038/s41592-020-01008-z}
}

@article{gu2023mamba,
  title   = {Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author  = {Gu, Albert and Dao, Tri},
  journal = {arXiv preprint arXiv:2312.00752},
  year    = {2023}
}

@inproceedings{liu2024vmamba,
  title     = {VMamba: Visual State Space Model},
  author    = {Liu, Yue and Tian, Yunjie and Zhao, Yuzhong and Yu, Hongtian and Xie, Lingxi and Wang, Yaowei and Ye, Qixiang and Liu, Yunfan},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {37},
  year      = {2024}
}

@inproceedings{rahman2024emcad,
  title     = {EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation},
  author    = {Rahman, Md Mostafijur and Munir, Mustafa and Marculescu, Radu},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {11769--11779},
  year      = {2024}
}

@inproceedings{shit2021cldice,
  title     = {clDice—a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation},
  author    = {Shit, Suprosanna and Paetzold, Johannes C. and Sekuboyina, Anjany and Ezhov, Ivan and Unger, Alexander and Zhylka, Andrey and Pluim, Josien P. W. and Bauer, Ulrich and Menze, Bjoern H.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {16560--16569},
  year      = {2021}
}
```

## Release checklist

- [x] Remove the EMCAD-derived decoder from the public source tree and replace
      it with the independently implemented `PSPADecoder3D`.
- [ ] Confirm the authorship/provenance of `LoE.py`, `MFCA.py`, `UnetMix.py`,
      and the 3-D selective-scan adaptations.
- [ ] Preserve upstream copyright and license notices in adapted source files.
- [ ] Add the PSPA-DT paper citation, DOI, authors, and institutional affiliation.
- [ ] Confirm that dataset and pretrained-weight licenses allow publication.
- [ ] Re-run this audit against the exact commit used for the public release.

Last audited: 7 September 2026.
