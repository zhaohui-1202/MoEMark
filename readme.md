# MoEMark

本地保留一套主模型代码，在原训练服务器运行。评测配置为**普通登记端 + Top-2 多专家检测端 + 图像绑定 + 几何类别绑定**，使用 6 个北京评价图层、20 张版权图像及 42、73、101 三个检测器种子。

GitHub 仓库包含代码、评测协议、依赖清单和本说明。模型权重、地图原始数据、版权图像、密钥、实验结果及论文文件不随仓库上传。下方目录中的 `结果/` 和 `论文/` 为本地已有目录。

## 目录

```text
代码/
  运行评测.py          统一入口：检查、评测、登记、验证、绘图
  绘制结果.py          NC/BCR 折线图和两种可区分性热图
  requirements.txt     Python 依赖
  评测协议/single92.json
  主模型/
    models.py          普通登记编码器、多专家检测器
    pipeline.py        权重加载、特征提取、登记与验证
    evaluate.py        鲁棒性及可区分性评测
    image_binding.py
    map_geometry_binding.py
    shared/            原实现中必需的几何处理、攻击和冻结编码器
结果/                  已有实验结果，以及重新绘图的输出
论文/                  已有论文文件
```

对比方法、消融、试验性变体和旧汇总脚本已从 `代码` 中移除。`shared` 保留部分原模块名以维持依赖关系，但不含原来的实验运行入口。服务器权重路径中的 `ablation` 是既有目录名，当前读取的是最终模型的权重，不会运行消融实验。

## 环境与资产

完整评测使用原服务器的 Linux、CUDA 和 PyTorch 环境。沿用已安装且与 CUDA 匹配的 PyTorch/torchvision；缺少其他依赖时参考 `代码/requirements.txt` 补齐。几何处理需要 Shapely 2。仅绘图需要 NumPy 和 Matplotlib，不需要 GPU。

默认资产根目录：

```text
/root/autodl-tmp/vector_zw_20261020
```

需要保留该目录下已有的地图及版权图像、原绑定密钥、干净地图特征缓存、图像特征库、共享几何编码器、图像编码器和三个种子的最终模型。代码不包含这些资产，也不会重新生成密钥或训练权重。`--stage check` 会列出缺失文件。

把整个 `代码` 文件夹上传到服务器的独立目录，例如 `/root/autodl-tmp/MoEMark/代码`；不要覆盖原模型资产目录。以下命令在该文件夹中执行。

## 运行

检查依赖与资产，不需要 GPU：

```bash
python 运行评测.py --stage check
```

开启 GPU 后，完整运行鲁棒性、可区分性并输出 PNG：

```bash
python 运行评测.py --stage all --workers 6
```

分别运行：

```bash
python 运行评测.py --stage robustness --workers 6
python 运行评测.py --stage discriminability
```

先用一个攻击设置检查运行环境：

```bash
python 运行评测.py --stage robustness --case-ids 1 --workers 2
```

局部评测输出 `robustness_subset.json/csv`，不会被标记为完整 92 档结果。随后执行完整命令可以继续完成剩余设置。默认输出到资产根目录的 `results/moemark_final_evaluation`；可通过 `--root` 指定资产根目录，通过 `--output` 指定新的结果目录。程序检查输入与模型标识后复用已完成的逐条记录，不覆盖原历史结果目录。

主要输出：

- `robustness.json/csv`：92 档攻击的 NC、BCR 和样本数。
- `records/`：逐条攻击记录、各种子得分及无效几何说明。
- `discriminability.json`：6×6 地图矩阵、20×20 版权图像矩阵。
- `figures/robustness_nc_bcr.png`：同图显示 NC 与 BCR。
- `figures/map_discriminability.png`、`figures/copyright_image_discriminability.png`：可区分性热图。

NC 为恢复的二值水印与参考水印的余弦相似度，BCR 为正确比特比例。载荷处理为灰度、NEAREST 16×16、阈值 128。鲁棒性先对重复和图像/种子求均值，再对各设置内的有效图层等权平均，最后对 92 个设置等权平均。无效几何记录单独保留。攻击 `source_id`、原语料排序及随机种子规则保持不变。

## 不开 GPU 重画已有结果

服务器已有本套评测输出时：

```bash
python 运行评测.py --stage plots
```

在本地项目根目录，读取已有论文评测结果绘图：

```powershell
python 代码/运行评测.py --stage plots --published-results 结果 --output 结果/主模型_重新绘图
```

这条命令只读取 `01_主模型鲁棒性` 和 `02_版权图像与地图区分`，不依赖对比实验或消融结果，也不会重新计算模型得分。图像均为 PNG。

## 单个地图的登记与验证

在服务器开启 GPU 后，使用同一种子的模型和同一份登记记录：

```bash
python 运行评测.py --stage register --seed 42 --map /path/original.shp --image /path/copyright.png --record /path/registration.json
python 运行评测.py --stage verify --seed 42 --map /path/candidate.shp --image /path/copyright.png --record /path/registration.json
```

验证返回 NC、BCR、匹配比特数和接受/拒绝结果，并检查模型、图像编码器、绑定协议及载荷版本。

## 本次验证范围

已完成语法和本地依赖检查、登记编码器与检测器的 CPU 数值一致性检查、共享地图表征一致性检查，并实际重画了已有结果的三张 PNG。完整的服务器 CUDA 评测尚未重跑；开启 GPU 后先执行上述单设置命令，再运行全量评测。
