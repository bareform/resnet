## ResNet

### Table of Contents
1. [Architecture](#Architecture)
1. [Dataset](#Dataset)
1. [Training](#Training)
1. [References](#References)

### Architecture
Residual networks (ResNets) are deep convolutional neural networks that introduced residual connections after each pair of 3x3 filters [<a href="#he2016resnet">1</a>]. These connections allow input tensors to bypass intermediate layers, helping to preserve the gradient flow and mitigate the vanishing gradient problem during backpropagation. ResNets come in five standard sizes: **ResNet18**, **ResNet34**, **ResNet50**, **ResNet101**, and **ResNet 152**.

<table>
<thead>
<tr>
<th>Layer Name</th>
<th>Output Size</th>
<th>18-layer</th>
<th>34-layer</th>
<th>50-layer</th>
<th>101-layer</th>
<th>152-layer</th>
</tr>

<tr>
<th>
  conv1
</th>
<td>

```math
32 \times 32
```

</td>
<td colspan="5" style="text-align: center;">

```math
3 \times 3,\, 64,\, \text{stride}\,1
```

</td>
</tr>

<tr>
<th>
  conv2_x
</th>
<td>

```math
32 \times 32
```

</td>
<td>

```math
\begin{bmatrix}
  3 \times 3, 64 \\
  3 \times 3, 64
\end{bmatrix} \times 2
```

</td>
<td>

```math
\begin{bmatrix}
  3 \times 3, 64 \\
  3 \times 3, 64
\end{bmatrix} \times 3
```

</td>
<td>

```math
\begin{bmatrix}
  1 \times 1, 64 \\
  3 \times 3, 64 \\
  1 \times 1, 256
\end{bmatrix} \times 3
```

</td>
<td>

```math
\begin{bmatrix}
  1 \times 1, 64 \\
  3 \times 3, 64 \\
  1 \times 1, 256
\end{bmatrix} \times 3
```

</td>
<td>

```math
\begin{bmatrix}
  1 \times 1, 64 \\
  3 \times 3, 64 \\
  1 \times 1, 256
\end{bmatrix} \times 3
```

</td>
</tr>

<tr>
<th>
  conv3_x
</th>
<td>

```math
16 \times 16
```

</td>
<td>

```math
\begin{bmatrix}
  3 \times 3, 128 \\
  3 \times 3, 128
\end{bmatrix} \times 2
```

</td>
<td>

```math
\begin{bmatrix}
  3 \times 3, 128 \\
  3 \times 3, 128
\end{bmatrix} \times 4
```

</td>
<td>

```math
\begin{bmatrix}
  1 \times 1, 128 \\
  3 \times 3, 128 \\
  1 \times 1, 512
\end{bmatrix} \times 4
```

</td>
<td>

```math
\begin{bmatrix}
  1 \times 1, 128 \\
  3 \times 3, 128 \\
  1 \times 1, 512
\end{bmatrix} \times 4
```

</td>
<td>

```math
\begin{bmatrix}
  1 \times 1, 128 \\
  3 \times 3, 128 \\
  1 \times 1, 512
\end{bmatrix} \times 8
```

</td>
</tr>

<tr>
<th>
  conv4_x
</th>
<td>

```math
8 \times 8
```

</td>
<td>

```math
\begin{bmatrix}
  3 \times 3, 256 \\
  3 \times 3, 256
\end{bmatrix} \times 2
```

</td>
<td>

```math
\begin{bmatrix}
  3 \times 3, 256 \\
  3 \times 3, 256
\end{bmatrix} \times 6
```

</td>
<td>

```math
\begin{bmatrix}
  1 \times 1, 256 \\
  3 \times 3, 256 \\
  1 \times 1, 1024
\end{bmatrix} \times 6
```

</td>
<td>

```math
\begin{bmatrix}
  1 \times 1, 256 \\
  3 \times 3, 256 \\
  1 \times 1, 1024
\end{bmatrix} \times 23
```

</td>
<td>

```math
\begin{bmatrix}
  1 \times 1, 256 \\
  3 \times 3, 256 \\
  1 \times 1, 1024
\end{bmatrix} \times 36
```

</td>
</tr>
<tr>
<th>
  conv5_x
</th>
<td>

```math
4 \times 4
```

</td>
<td>

```math
\begin{bmatrix}
  3 \times 3, 512 \\
  3 \times 3, 512
\end{bmatrix} \times 2
```

</td>
<td>

```math
\begin{bmatrix}
  3 \times 3, 512 \\
  3 \times 3, 512
\end{bmatrix} \times 3
```

</td>
<td>

```math
\begin{bmatrix}
  1 \times 1, 512 \\
  3 \times 3, 512 \\
  1 \times 1, 2048
\end{bmatrix} \times 3
```

</td>
<td>

```math
\begin{bmatrix}
  1 \times 1, 512 \\
  3 \times 3, 512 \\
  1 \times 1, 2048
\end{bmatrix} \times 3
```

</td>
<td>

```math
\begin{bmatrix}
  1 \times 1, 512 \\
  3 \times 3, 512 \\
  1 \times 1, 2048
\end{bmatrix} \times 3
```

</td>
</tr>

<tr>
<td></td>
<td>

```math
1 \times 1
```

</td>
<td colspan="5" style="text-align: center;">

```math
2 \times 2 \,\, \text{average pool},\, \text{fully connected layer},\, \text{softmax}
```

</td>
</tr>

</thead>
</table>

### Dataset
You can download the CIFAR-10 dataset [<a href="#krizhevskyt2009cifar10">2</a>] using the following command:

```console
python3 data/download_cifar10.py
```

This command will download place the dataset at the following path: `./data/datasets`.

The dataset consist of 60000 32x32 color images split into 10 classes, with 6000 images per class. There are a total of 50000 training images and 10000 test images. The classes (in alphabetically order) in the dataset are:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

The classes are completely mutually exclusive (i.e. there is no overlap between automobiles and trucks).

### Training
You can train a ResNet18 model on CIFAR 10 using the following command:

```console
python3 -m utils.trainer \
  --root './data/datasets' \
  --batch_size 32 \
  --num_workers 1 \
  --num_epochs 20 \
  --lr 0.001 \
  --optimizer 'AdamW' \
  --lr_scheduler 'CosineAnnealingLR' \
  --t_max 10 \
  --model_name 'ResNet18' \
  --random_seed 0 \
  --ckpt_dir './checkpoints' \
  --compile \
  --pin_memory \
  --transform
```

To train a different size ResNet model change the value of the `--model_name` argument. For example, to train a ResNet50 model the new command will be:

```console
python3 -m utils.trainer \
  --root './data/datasets' \
  --batch_size 32 \
  --num_workers 1 \
  --num_epochs 20 \
  --lr 0.001 \
  --optimizer 'AdamW' \
  --lr_scheduler 'CosineAnnealingLR' \
  --t_max 10 \
  --model_name 'ResNet50' \
  --random_seed 0 \
  --ckpt_dir './checkpoints' \
  --compile \
  --pin_memory \
  --transform
```

### References
<a name="he2016resnet"></a>[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pages 770–778, 2016.

<a name="krizhevskyt2009cifar10"></a>[2] Alex Krizhevsky and Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. *Master’s thesis, University of Toronto*, 2009.
