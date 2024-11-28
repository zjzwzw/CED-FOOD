# Conditional-Evidence-Decoupling-FOOD

[Few-Shot Open-Set Object Detection via Conditional Evidence Decoupling](https://arxiv.org/abs/2406.18443 "Few-Shot Open-Set Object Detection via Conditional Evidence Decoupling")

Few-shot Open-set Object Detection (FOOD) poses a significant challenge in real-world scenarios. It aims to train an open-set detector under the condition of scarce training samples, which can detect known objects while rejecting unknowns. Under this challenging scenario, the decision boundaries of unknowns are difficult to learn and often ambiguous. To mitigate this issue, we develop a two-stage open-set object detection framework with prompt learning, which delves into conditional evidence decoupling for the unknown rejection. Specifically, we propose an Attribution-Gradient-based Pseudo-unknown Mining (AGPM) method to select region proposals with high uncertainty, which leverages the discrepancy in attribution gradients between known and unknown classes, alleviating the inadequate unknown distribution coverage of training data. Subsequently, we decouple known and unknown properties in pseudo-unknown samples to learn distinct knowledge with proposed Conditional Evidence Decoupling (CED), which enhances separability between knowns and unknowns. Additionally, we adjust the output probability distribution through Abnormal Distribution Calibration (ADC), which serves as a regularization term to establish robust decision boundaries for the unknown rejection. Our method has achieved superior performance over previous state-of-the-art approaches, improving the mean recall of unknown class by 7.24% across all shots in VOC10-5-5 dataset settings and 1.38% in VOC-COCO dataset settings.

- ## **Visualization**

![visualized](visualized.png)

- ## **Installation**

```bash
git clone https://github.com/zjzwzw/CED-FOOD.git
cd CED-FOOD/

conda create -n CED-FOOD python=3.8 -y
conda activate CED-FOOD

pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

python -m pip install -e detectron2-0.3

pip install -r requirements.txt
```

- ## **Prepare datasets**

same as [HSIC-based Moving WeightAveraging for Few-Shot Open-Set Object Detection (ACM MM'23)](https://github.com/binyisu/food)

- ## Prepare models

Follow ".\offline_rpn_weights\README.md" and ".\pretrained_ckpt\regionclip\README.md" to prepare pretrained models.

- ## Running

  - ##### VOC-COCO dataset settings:

    ```bash
    bash voc-coco.sh
    ```

  - ##### VOC10-5-5 dataset settings:

    ```bash
    bash voc10-5-5.sh
    ```

  
  

Note that the comm.py, rpn.py, proposal_utils.py and batch_norm.py are modified version based on the [Release v0.3 · facebookresearch/detectron2 (github.com)](https://github.com/facebookresearch/detectron2/releases/tag/v0.3)

All our experiments were conducted on a single NVIDIA 1080Ti, with a batch size of 1 for base class training and a batch size of 1 for novel class fine-tuning.
