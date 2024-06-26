# Conditional-Evidence-Decoupling-FOOD



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

    

