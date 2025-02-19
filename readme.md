# StainPresetNet

## Abstract

Stain normalization can reduce the color differences affected by various factors, thereby improving the performance of computer-aided diagnostic systems. Conventional stain normalization methods extract mapping relationships from one or several reference images and perform normalization on a pixel-by-pixel manner. Conventional methods can flexibly adjust the style of the output image, but these methods are difficult to extract accurate color mapping relationships from the reference image. Although existing deep learning-based stain normalization methods can accurately extract color mapping relationships from entire datasets, these methods often employ complex deep neural networks, resulting in low computational efficiency and artifact risks. In addition, the normalization direction of existing deep learning-based methods is usually fixed, and when the normalization direction needs to be changed, retraining is needed to learn a new color-mapping relationship. In this paper, we proposed a novel stain normalization method named StainPresetNet, which achieves structure preservation, accurate mapping relationships, multi-domain to multi-domain applicability, and high computational efficiency. StainPresetNet extracts color mapping relationships from entire datasets, normalizes on a pixel-by-pixel manner, and utilizes a preset reference image to guide the normalization direction, meanwhile maintaining high computational efficiency. The results on cytopathology and histopathology datasets demonstrate that StainPresetNet achieves accurate color mapping relationships and can improve the generalization of classifiers on pathology diagnosis tasks.

## Model Architecture

![StainPresetNet](./assets/StainPresetNet.jpg)

## Transform between multiple centers

|      |                              C1                              |                              C2                              |                              C3                              |                              C4                              |                              C5                              |
| ---- | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| C1   | ![C1_85_source_7328](./result/C1_85_source_7328/C1_85_source_7328.png) | ![C2_46_source_12318](./result/C1_85_source_7328/C2_46_source_12318.png) | ![C3_28_source_15437](./result/C1_85_source_7328/C3_28_source_15437.png) | ![C4_60_source_23330](./result/C1_85_source_7328/C4_60_source_23330.png) | ![C5_25_source_29176](./result/C1_85_source_7328/C5_25_source_29176.png) |
| C2   | ![C1_85_source_7328](./result/C2_46_source_12318/C1_85_source_7328.png) | ![C2_46_source_12318](./result/C2_46_source_12318/C2_46_source_12318.png) | ![C3_28_source_15437](./result/C2_46_source_12318/C3_28_source_15437.png) | ![C4_60_source_23330](./result/C2_46_source_12318/C4_60_source_23330.png) | ![C5_25_source_29176](./result/C2_46_source_12318/C5_25_source_29176.png) |
| C3   | ![C1_85_source_7328](./result/C3_28_source_15437/C1_85_source_7328.png) | ![C2_46_source_12318](./result/C3_28_source_15437/C2_46_source_12318.png) | ![C3_28_source_15437](./result/C3_28_source_15437/C3_28_source_15437.png) | ![C4_60_source_23330](./result/C3_28_source_15437/C4_60_source_23330.png) | ![C5_25_source_29176](./result/C3_28_source_15437/C5_25_source_29176.png) |
| C4   | ![C1_85_source_7328](./result/C4_60_source_23330/C1_85_source_7328.png) | ![C2_46_source_12318](./result/C4_60_source_23330/C2_46_source_12318.png) | ![C3_28_source_15437](./result/C4_60_source_23330/C3_28_source_15437.png) | ![C4_60_source_23330](./result/C4_60_source_23330/C4_60_source_23330.png) | ![C5_25_source_29176](./result/C4_60_source_23330/C5_25_source_29176.png) |
| C5   | ![C1_85_source_7328](./result/C5_25_source_29176/C1_85_source_7328.png) | ![C2_46_source_12318](./result/C5_25_source_29176/C2_46_source_12318.png) | ![C3_28_source_15437](./result/C5_25_source_29176/C3_28_source_15437.png) | ![C4_60_source_23330](./result/C5_25_source_29176/C4_60_source_23330.png) | ![C5_25_source_29176](./result/C5_25_source_29176/C5_25_source_29176.png) |



## Install

```bash
pip install requirements.txt
```

## Train

Save images inside the folders in the format as shown below:

```
train
   ├── Domain 1
   |   ├── a.jpg  (name doesn't matter)
   |   ├── b.jpg
   |   └── ...
   ├── Domain 2
   |   ├── c.jpg
   |   ├── d.jpg
   |   └── ...
```

```python
python main.py --name [Your Exp Name] --train_dir_root [Path to data]
```

## Test

```python
python test.py
```

