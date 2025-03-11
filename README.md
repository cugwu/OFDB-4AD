# One-instance fractals for anomaly detection

This repository implements the One-instance FractalDataBase (OFDB) [1], and apply it for defect detection. 
OFDB is a dataset of abstract fractal images, with only one sample per class.  

We generate a dataset with 1,000 classes to train a ResNet-like backbones and leverage the pre-trained features for defect detection. 
Despite using only 1,000 samples, strong data augmentation allows the model to achieve results comparable to those pre-trained on ImageNet, 
which requires over a million samples and extensive training time.

## Getting Started

The code uses [PyTorch](https://pytorch.org/) for training. Follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) 
to install the version compatible with your system. Additional required packages are listed in `requirements.txt`.
```bash
# clone project
git clone https://github.com/cugwu/OFDB-4AD.git
   
# [RECOMMENDED] set up a virtual environment
python -m venv venv_name  # choose your prefered venv_name
source venv_name/bin/activate

# install requirements
pip install -r requirements.txt
```

## Dataset Generation
The dataset used for training is in `./1p-fractals`. For the dataset generation we follow [fractal4AD](https://github.com/cugwu/fractal4AD) repository.
The repository contains code for generating large-scale datasets of fractal images.

First an Iterated Function Systems (IFS), which serves as the basis for creating the final fractals dataset need to be generated
```bash
# generate a dataset of 100,000 systems, each with between 2 and 4 affine transformations
python ifs.py --save_path ./ifs-100k.pkl --num_systems 100000 --min_n 2 --max_n 4
```
This will produce a `.pkl` file containing a dictionary with the following structure:
```python
{
  "params": [
    {"system": np.array(...)},
    ...
  ],
  "hparams": {
    ...
  }
}
```

The generated `.pkl` file can be used as an input parameter to generate the final fractals dataset. The following command 
generates a fractal dataset without a background, using the IFS dataset stored in `ifs-100k.pkl`. From this IFS database, 
50,000 IFS samples will be used to create a dataset with 1,000 classes, each containing 1 sample of size 512x512:

```bash
python dataset_generator.py --dataset_type fractals --param_file ./ifs-100k.pkl --dataset_dir ./fractals_dataset --num_systems 50000 --num_class 1000 --per_class 1 --size 512
```

The dataset can also be generated directly without the `.pkl` file. In this case, the `ifs.py` script will be called internally by the `dataset_generator.py` script:
```bash
python dataset_generator.py --dataset_type fractals --dataset_dir ./fractals_dataset --num_systems 50000 --num_class 1000 --per_class 1 --size 512
```

## Training
For training use the following commands:
```bash
python main.py --datadir ./1p-fractals --outdir ./fractals_pretrain  --store_name fractals_resnet50 --arch resnet50 --num_class 1000 --batch_size 128 --epochs 20000
```
We also offer additional backbone architectures: Fast Fourier Convolution [2] and Global Filter Networks [3].
Users can select their preferred model by setting the `--arch` parameter. The full list of available models can be found in `networks.py`.

## Anomaly Detection Results
We provide the results on ResNet-50 and WideResNet-50 models pre-trained on ImageNet and OFDB-1k, as anomaly detection method 
we use the original code of [PatchCore](https://github.com/amazon-science/patchcore-inspection). In the table are reported
respectively: instance_auroc, full_pixel_auroc, and anomaly_pixel_auroc.

| MVTec PatchCore | resnet50           | wide_resnet50     |
|-----------------|--------------------|-------------------|
| IN              | 0.989/0.981/0.973  | 0.991/0.981/0.973 |
| OFDB-1k         | 0.937/0.965/0.956  | 0.936/0.959/0.949 |


## License
This project is distributed under the Apache License, Version 2.0. See `LICENSE.txt` for more details.
Some parts of the code are licensed under the MIT License. For those cases, the MIT License text is included at the top 
of the respective files.

## Citation
If you use this code or find our work helpful for your resarch, please cite the following papers:
```bibtex
@misc{OFDB-4AD,
  author = {cugwu},
  title = {One-instance fractals for anomaly detection},
  year = {2025},
  url = {https://github.com/cugwu/OFDB-4AD},
  note = {Accessed: 2025-03-11}
}
```

## References
[1] Nakamura, R., Kataoka, H., Takashima, S., Noriega, E. J. M., Yokota, R., & Inoue, N. (2023). Pre-training vision transformers with very limited synthesized images. 
In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 20360-20369).

[2] Chi, L., Jiang, B., & Mu, Y. (2020). Fast fourier convolution. Advances in Neural Information Processing Systems, 33, 4479-4488.

[3] Rao, Y., Zhao, W., Zhu, Z., Zhou, J., & Lu, J. (2023). GFNet: Global filter networks for visual recognition. 
IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(9), 10960-10973.
