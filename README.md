# AI for Cognitive Ultrasound Imaging
### [IEEE IUS 2026 Short Course](https://ieee-ius.org/2026/short-course/ai-for-cognitive-ultrasound-imaging)

**8:00 AM to 12:30 PM**

The AI landscape has changed fast: from computer vision and CNNs, to **deep generative modeling** and **cognitive agents** (active inference and RL). This short course brings those breakthroughs straight to ultrasound, with theory *and* hands-on code.

We'll cover:

- 🌀 **Deep generative AI** (diffusion and flow matching), from theory to implementation
- 🤖 **Active inference and perception-action loops**, from theory to implementation
- 🩺 **Ultrasound applications**, from cardiac image dehazing to adaptive scanning
- ⚡ **Real-time demos** on the US4US open ultrasound platform

## 👩‍🏫 Organizers

- [Yonina Eldar](https://ieee-ius.org/2026/contacts/yonina-eldar), Northeastern University
- [Marcin Lewandowski](https://ieee-ius.org/2026/contacts/marcin-lewandowski), us4us, Ltd.
- [Ruud van Sloun](https://ieee-ius.org/2026/contacts/ruud-van-sloun), Eindhoven University of Technology

The hands-on notebook portion of this course is brought to you by [Tristan Stevens](https://github.com/tristan-deep), [Oisín Nolan](https://github.com/OisinNolan) and [Wessel van Nierop](https://github.com/wesselvannierop), maintainers of [`zea`](https://github.com/tue-bmd/zea) 🚀.

## 🚀 Get started

Open the notebook directly in Google Colab (free GPU included), no local setup needed:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tue-bmd/ius-2026-cogus-workshop/blob/main/ius_2026_cogus_workshop.ipynb)

Once opened, enable a GPU via **Runtime → Change runtime type → GPU**, then run the cells top to bottom. The notebook is fully interactive: tweak the settings, swap the action-selection strategy, and see what happens.

## 🔬 What you'll build

A basic **perception-action loop** for adaptive ultrasound imaging, powered by a [flow matching](https://zea.readthedocs.io/en/latest/_autosummary/zea.models.flow_matching.html) generative model and greedy entropy-based transmit steering, all built on [`zea`](https://github.com/tue-bmd/zea), the open-source library for AI-powered ultrasound.

## 🛠️ Setup

> A GPU is strongly advised, this notebook can be slow on CPU only.

### Colab

Just click the badge above, no setup needed. Enable a GPU via **Runtime → Change runtime type → GPU**.

### Local

This repo uses [`uv`](https://docs.astral.sh/uv/) for a quick, reproducible setup:

```bash
git clone https://github.com/tue-bmd/ius-2026-cogus-workshop.git
cd ius-2026-cogus-workshop
uv sync
uv run jupyter notebook ius_2026_cogus_workshop.ipynb
```

## 📚 References

- Ruud J. G. van Sloun. Active inference and deep generative modeling for cognitive ultrasound. *IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control*, 71(11):1478–1490, 2024. doi:[10.1109/TUFFC.2024.3466290](https://doi.org/10.1109/TUFFC.2024.3466290).
- Tristan S.W. Stevens, Wessel L. van Nierop, Ben Luijten, Vincent van de Schaft, Oisín Nolan, Beatrice Federici, Louis D. van Harten, Simon W. Penninga, Noortje I.P. Schueler, and Ruud J.G. van Sloun. zea: a toolbox for cognitive ultrasound imaging. *Journal of Open Source Software*, 11(121):9881, 2026. doi:[10.21105/joss.09881](https://doi.org/10.21105/joss.09881).
- Wessel L. van Nierop, Oisín Nolan, Tristan S. W. Stevens, and Ruud J. G. van Sloun. Patient-adaptive echocardiography using cognitive ultrasound. *IEEE Trans. Medical Imaging*, 45(7):4034–4046, 2026. doi:[10.1109/tmi.2026.3691009](https://doi.org/10.1109/tmi.2026.3691009).

See you at IUS 2026! 🎉
