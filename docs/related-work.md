# Related Work

Lane line detection has attracted sustained attention from the computer vision and autonomous driving communities. Approaches fall into two broad families: traditional signal-processing methods, and deep learning methods.

---

## Traditional Methods

Early systems relied on hand-crafted geometric and colour features.

**Kalman filter + ROI (Chen et al., 2016):** Dynamic identification of a Region of Interest (ROI) parallel to lane lines using a Kalman filter, reducing computation and enabling real-time operation. Fragile under partial occlusion and changing lighting.

**Vanishing point + edge distribution (Wang et al., 2014):** Lane detection using curvature features of near and far fields, leveraging the vanishing point and edge distribution functions. Sensitive to road surface texture changes.

**YCbCr + Hough transform (Wu et al., 2020):** Yellow and white lane lines extracted from the YCbCr colour space, then located with the Hough transform. Works well on clear, well-marked roads; fails on shadows and wet surfaces.

**Shared limitations:** These methods rely on stable illumination, clean markings, and structured road geometry. Accuracy drops severely in low-light conditions, partial occlusions, or faded markings — making them insufficient for reliable autonomous driving.

---

## Deep Learning Methods

CNNs replaced hand-crafted features with hierarchical learned representations, dramatically improving robustness.

### Semantic and Instance Segmentation

**LaneNet (Neven et al., 2018):** Reframed lane detection as an instance segmentation task, allowing an arbitrary number of lane lines to be distinguished individually. End-to-end trainable; set the standard for binary segmentation of lane pixels.

**Dual-view CNN (He et al., 2016):** Separate network branches for near-field and far-field texture, reducing false detections caused by gradient textures on the road surface.

### Spatio-Temporal and Recurrent Models

**ConvGRU network (Zhang et al., 2022):** Integrated Convolutional Gated Recurrent Units (ConvGRUs) to process consecutive video frames, modelling the temporal dynamics of lane lines. Effective for occlusion recovery across frames.

**CNN + RNN (Zou et al., 2020):** Hybrid architecture learning features from consecutive frames via a recurrent backbone. Reduces false detections in complex scenes by propagating temporal context.

**VPGNet (Lee et al., 2017):** Vanishing Point Guided Network that incorporates vanishing point supervision during training, improving performance in low-light and rain conditions.

### Spatial Propagation Networks

**SCNN (Pan et al., 2018):** Spatial CNN propagates information between adjacent pixels slice-by-slice, recovering occluded lane lines that CNN with local receptive fields would miss. Highly accurate but computationally expensive — pixel-to-pixel message passing limits throughput on embedded hardware.

**RiLLD-Net + GAN (Zhang et al., 2021):** Combines ripple-based lane detection with Wasserstein GANs to hallucinate plausible lane markings in occluded regions. Attempts to ease the compute burden of SCNN while maintaining accuracy.

---

## Motivation for ST-LaneNet

The state of the art reveals a persistent tension between three desiderata:

1. **Accuracy under no-visual-cue conditions** (occlusion, poor lighting) — requires global context.
2. **Computational efficiency** — required for real-time deployment on embedded systems.
3. **Robustness** — hand-crafted IPM assumptions limit cross-domain transfer.

SCNN solves (1) but fails (2). Traditional methods satisfy (2) but fail (1). ST-LaneNet addresses the gap by combining:

- A **lightweight edge proposal branch** (depthwise-separable convolutions + CARAFE upsampling) operating in bird's-eye view for precise, compute-efficient local structure detection.
- A **Swin Transformer backbone** for global context at O(n) attention complexity — replacing the quadratic cost of standard self-attention with window-local attention and shifted-window cross-window communication.

→ See [Architecture](architecture.md) for the full structural description.

---

## References

| Key | Citation |
|-----|----------|
| du2024 | Du et al., *Chinese Journal of Mechanical Engineering*, vol. 37, 2024 |
| chen2016 | Chen et al., *Automotive Engineering*, vol. 38(2), 2016 |
| wang2014 | Wang et al., *Journal of Traffic and Transportation Engineering*, vol. 14(5), 2014 |
| wu2020 | Wu & Zhao, *China Journal of Highway and Transport*, vol. 33(5), 2020 |
| neven2018 | Neven et al., *IEEE IV Symposium*, 2018 |
| he2016 | He et al., *IEEE IV Symposium*, 2016 |
| zhang2022 | Zhang et al., *IEEE TITS*, vol. 23(7), 2022 |
| zou2020 | Zou et al., *IEEE TVT*, vol. 69(1), 2020 |
| lee2017 | Lee et al., *IEEE ICCV*, 2017 |
| pan2018 | Pan et al., *AAAI*, 2018 |
| zhang2021 | Zhang et al., *IEEE TITS*, vol. 22(3), 2021 |
