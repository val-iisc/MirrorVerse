<div align="center">

<h1>MirrorVerse: Pushing Diffusion Models to Realistically Reflect the World</h1>
<h1>CVPR 2025</h1>

<p align="center">
    <a href="https://www.linkedin.com/in/ankit-dhiman-46109a174/" target="_blank"><strong>Ankit Dhiman</strong></a> <sup>1,2<b>*</b></sup>
    ·
    <a href="https://cs-mshah.github.io/" target="_blank"><strong>Manan Shah</strong></a> <sup>1<b>*</b></sup>
    ·
    <a href="https://cds.iisc.ac.in/faculty/venky/" target="_blank"><strong>R Venkatesh Babu</strong></a> <sup>1</sup>
</p>

<p align="center">
    <sup><b>*</b></sup> Equal Contribution <br>
    <sup>1</sup> Vision and AI Lab, IISc Bangalore <br>
    <sup>2</sup> Samsung R & D Institute India - Bangalore
</p>

<a href="https://arxiv.org/abs/2504.15397">
<img src='https://img.shields.io/badge/arxiv-MirrorVerse-red' alt='Paper PDF'></a>
<a href="https://mirror-verse.github.io/">
<img src='https://img.shields.io/badge/Project-Website-green' alt='Project Page'></a>
<a href="https://huggingface.co/datasets/ankitIIsc/SynMirrorV2">
<img src='https://img.shields.io/badge/Dataset-HuggingFace-blue' alt='Dataset'></a>
<a href="https://github.com/val-iisc/Reflecting-Reality">
<img src='https://img.shields.io/badge/Previous Work-Reflecting Reality-9cf' alt='Reflecting Reality'></a>

<br>
<img src='assets/teaser.jpg' alt='MirrorVerse Teaser' height='100%' width='100%'>

</div>

---

## 🧠 Overview

**MirrorVerse** builds upon our prior work *Reflecting Reality*, pushing the frontier of mirror reflection generation by adding diversity in the synthetic dataset creation pipeline and leveraging curriculum learning for generalizing to real-world scenes.

We introduce **SynMirrorV2**, a large-scale synthetic dataset containing **207K** samples with full scene geometry, including depth maps, normal maps, and segmentation masks. **SynMirrorV2** has high-fidelity training samples featuring variable object poses, occlusions, and multi-object setups.


---

## 🚀 Highlights

- 📦 **SynMirrorV2 Dataset**: 207K synthetic samples with diverse object configurations and camera poses.
- 🧩 **Curriculum Learning Strategy**: a curriculum learning strategy that progressively adapts to complex scenarios, enabling state-of-the-art model to generalize better to real-world reflections.
- 🖼️ **Multi-object Reflection Generation**: First approach to effectively handle complex multi-object mirror scenes.
- 📊 **Robust Benchmarks**: Demonstrates strong quantitative and qualitative gains over previous SOTA.

---

## 🗓️ TODO

- [X] [14/07/2025] 🔥 ~~Release the SynMirrorV2 Dataset~~
- [X] Release 🔥 ~~checkpoints trained on SynMirrorV2~~ [Link](#-checkpoint-details)
- [X]  [07/06/2025] 🔥 ~~Release codebase for creating synthetic dataset~~ [Link](BlenderProc/reflection/README.md)
- [] Add interactive notebook demo for inference

---

## 💾 Checkpoint Details

The following table summarizes the key checkpoints mentioned in the project, along with their links and descriptions.

| Checkpoint Name                                  | Link                                                                  | Description                                                                                                                                                                                                                                                                                                                         |
| :----------------------------------------------- | :----------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MirrorFusion-v2** | [Google Drive](https://drive.google.com/drive/folders/1T0ldC8xIo4Z-LJ0em5SkpZAYiI49TjcC?usp=sharing) | This checkpoint is trained on single and multiple objects from [SynMirrorV2](https://huggingface.co/datasets/ankitIIsc/SynMirrorV2).                                                                                                          |                                 |
| **MirrorFusion-v2-MSD** | [Google Drive](https://drive.google.com/drive/folders/1raz52DndBbkyEIIQn1IFUA_56tgYCagz?usp=sharing) | This checkpoint is finetuned on real-world [MSD](https://mhaiyang.github.io/ICCV2019_MirrorNet/index) dataset.                                                                                                          |                                 |

## 🤝🏼 Citation
```
@inproceedings{dhiman2025mirrorverse,
  title={MirrorVerse: Pushing Diffusion Models to Realistically Reflect the World},
  author={Dhiman, Ankit and Shah, Manan and Babu, R Venkatesh},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={11239--11249},
  year={2025}
}
```

## 💖 Acknowledgements

This work builds on the foundation of [Reflecting Reality](https://github.com/val-iisc/Reflecting-Reality). We also thank the developers of [BlenderProc](https://github.com/DLR-RM/BlenderProc), [diffusers](https://github.com/huggingface/diffusers), and [SAM](https://github.com/facebookresearch/segment-anything) for their amazing tools and libraries.
