# 自动驾驶场景下多目标跟踪算法对比研究

## 项目简介

本项目针对自动驾驶场景下的多目标跟踪（Multiple Object Tracking, MOT）任务，对三种经典跟踪算法（SORT、DeepSORT、FairMOT）进行了全面的准确性分析和遮挡影响的量化研究。研究基于KITTI数据集，使用YOLOv8作为目标检测器，系统评估了不同算法在复杂驾驶场景中的性能表现。

## 研究目标

1. **算法性能对比**：系统评估SORT、DeepSORT和FairMOT三种算法在自动驾驶场景下的跟踪准确性
2. **遮挡影响分析**：量化分析遮挡对跟踪性能的影响，为算法选择提供依据
3. **指标评估**：使用MOT标准指标（MOTA、MOTP、IDF1等）进行全面的性能评估

## 项目结构

```
sort_deeprsort_fairmot/
├── data/                          # 数据目录
│   └── kitti/                     # KITTI数据集
│       ├── training/              # 训练集
│       │   ├── image_02/          # 图像序列
│       │   ├── det/               # YOLOv8检测结果
│       │   └── label_02/          # 真值标注
│       └── testing/               # 测试集
│
├── src/                           # 算法实现源码
│   ├── sort/                      # SORT算法实现
│   │   ├── sort.py                # SORT主类
│   │   └── kalman_filter.py       # 卡尔曼滤波器
│   ├── deepsort/                  # DeepSORT算法实现
│   │   ├── deepsort.py            # DeepSORT主类
│   │   ├── tracker.py             # 跟踪器
│   │   ├── kalman_filter.py       # 卡尔曼滤波器
│   │   ├── iou_matching.py        # IoU匹配
│   │   ├── linear_assignment.py   # 线性分配
│   │   ├── nn_matching.py         # 最近邻匹配
│   │   └── reid/                  # ReID模块
│   │       ├── model.py            # ReID模型
│   │       ├── extractor.py        # 特征提取器
│   │       └── ckpt.t7             # 预训练权重
│   └── fairmot/                   # FairMOT算法实现
│       ├── fairmot_model.py       # FairMOT模型
│       └── tracker.py             # 跟踪器
│
├── scripts/                       # 运行脚本
│   ├── run_sort.py                # 运行SORT跟踪
│   ├── run_deppsort.py            # 运行DeepSORT跟踪
│   └── run_fairmot.py             # 运行FairMOT跟踪
│
├── analysis/                      # 评估和分析脚本
│   ├── eval_kitti.py              # KITTI评估
│   ├── eval_compare.py            # 算法对比评估
│   ├── eval_deepsort.py           # DeepSORT详细分析
│   ├── evlal_fairmot.py           # FairMOT详细分析
│   └── visualize_tracks.py        # 可视化脚本
│
├── results/                        # 跟踪结果
│   ├── sort_results/              # SORT跟踪结果
│   ├── deepsort_result/           # DeepSORT跟踪结果
│   └── fairmot_result/            # FairMOT跟踪结果
│
├── results_analysis/              # 分析结果
│   ├── compare/                   # 算法对比分析
│   │   ├── mot_metrics_comparison.csv  # 对比指标
│   │   └── plots/                 # 可视化图表
│   ├── deepsort_metrics/          # DeepSORT详细指标
│   ├── fairmot_metrics/           # FairMOT详细指标
│   ├── metrics/                   # 通用指标
│   ├── gt_mot_format/             # 转换后的GT格式
│   └── vis/                       # 可视化结果
│
├── weights/                       # 模型权重
│   └── fairmot_dla34.pth          # FairMOT预训练权重
│
├── gen_det_yolov8.py              # YOLOv8检测生成脚本
├── utils.py                       # 工具函数
├── main.py                        # 主程序入口
└── yolov8n.pt                     # YOLOv8模型权重
```

## 环境配置

### 依赖要求

- Python 3.7+
- PyTorch 1.8+
- OpenCV
- NumPy
- Pandas
- motmetrics
- ultralytics (YOLOv8)
- scipy
- matplotlib (用于可视化)

### 安装步骤

1. **克隆项目**
```bash
git clone <repository_url>
cd sort_deeprsort_fairmot
```

2. **安装依赖**
```bash
pip install torch torchvision
pip install opencv-python numpy pandas scipy matplotlib
pip install motmetrics
pip install ultralytics
```

3. **准备数据**
   - 下载KITTI数据集并放置在 `data/kitti/` 目录下
   - 确保目录结构如下：
     ```
     data/kitti/training/
     ├── image_02/          # 图像序列
     ├── det/               # 检测结果（由gen_det_yolov8.py生成）
     └── label_02/          # 真值标注
     ```
   - 由于数据集较大请从官方下载：[KITTI官网]([链接地址](http://www.cvlibs.net/datasets/kitti/eval_tracking.php?benchmark=tracking))
      
4. **下载模型权重**
   - YOLOv8权重：`yolov8n.pt`（已包含在项目中）
   - DeepSORT ReID权重：`src/deepsort/reid/ckpt.t7`（已包含）
   - FairMOT权重：`weights/fairmot_dla34.pth`[[Google](https://drive.google.com/file/d/1SFOhg_vos_xSYHLMTDGFVZBYjo8cr2fG/view)][[Baidu](https://pan.baidu.com/share/init?surl=JZMCVDyQnQCa5veO73YaMw)]

## 使用流程

### 1. 生成检测结果

首先使用YOLOv8生成目标检测结果：

```bash
python gen_det_yolov8.py
```

该脚本会：
- 读取KITTI训练集的图像序列
- 使用YOLOv8进行目标检测
- 将检测结果保存到 `data/kitti/training/det/` 目录
- 输出格式：每帧一个txt文件，格式为 `x1 y1 x2 y2 score`

### 2. 运行跟踪算法

#### 运行SORT跟踪

```bash
python scripts/run_sort.py
```

#### 运行DeepSORT跟踪

```bash
python scripts/run_deppsort.py
```

#### 运行FairMOT跟踪

```bash
python scripts/run_fairmot.py
```

所有跟踪结果将保存在 `results/` 目录下对应的子目录中，格式为：
```
frame_id track_id x1 y1 x2 y2
```

### 3. 评估和对比

#### 算法对比评估

运行综合对比评估，生成MOT指标对比：

```bash
python analysis/eval_compare.py
```

该脚本会：
- 将KITTI真值转换为MOT格式
- 将三种算法的跟踪结果转换为MOT格式
- 计算MOT标准指标（MOTA、MOTP、IDF1、IDSW等）
- 生成对比CSV文件：`results_analysis/compare/mot_metrics_comparison.csv`

#### 单个算法详细评估

**评估SORT：**
```bash
python analysis/eval_kitti.py --model sort
```

**评估DeepSORT：**
```bash
python analysis/eval_deepsort.py
```

**评估FairMOT：**
```bash
python analysis/evlal_fairmot.py
```

### 4. 结果分析

分析结果保存在 `results_analysis/` 目录下：

- **compare/**: 三种算法的对比分析结果
  - `mot_metrics_comparison.csv`: 综合对比指标
  - `plots/`: 可视化对比图表
  
- **deepsort_metrics/**: DeepSORT详细分析
  - 各序列的统计指标
  - 相关性分析
  - 描述性统计
  
- **fairmot_metrics/**: FairMOT详细分析
  - 各序列的指标
  - 遮挡级别统计
  
- **metrics/**: SORT评估指标

### 5. 可视化

## 评估指标

本项目使用MOT标准评估指标：

- **MOTA (Multiple Object Tracking Accuracy)**: 综合跟踪准确度
- **MOTP (Multiple Object Tracking Precision)**: 跟踪精度
- **IDF1**: ID F1分数，衡量ID保持能力
- **IDSW (ID Switches)**: ID切换次数
- **FP (False Positives)**: 误检数量
- **FN (False Negatives)**: 漏检数量
- **Precision**: 精确率
- **Recall**: 召回率

## 算法特点

### SORT (Simple Online and Realtime Tracking)
- **特点**：简单高效，基于卡尔曼滤波和匈牙利算法
- **优势**：速度快，实时性好
- **劣势**：对遮挡和快速运动处理能力有限

### DeepSORT
- **特点**：在SORT基础上加入深度特征匹配
- **优势**：通过ReID特征提高ID保持能力，对遮挡更鲁棒
- **劣势**：计算开销较大

### FairMOT
- **特点**：端到端的联合检测和跟踪框架
- **优势**：检测和跟踪联合优化，性能优异
- **劣势**：模型复杂度高，需要更多计算资源

## 遮挡影响分析

项目包含对遮挡影响的量化分析：

1. **遮挡级别统计**：分析不同遮挡程度下的跟踪性能
2. **相关性分析**：研究遮挡与跟踪指标的相关性
3. **对比分析**：比较不同算法在遮挡场景下的表现

详细分析结果见 `results_analysis/` 目录下的相关文件。

## 实验结果

实验结果保存在 `results_analysis/` 目录下，包括：

- 各算法的MOT指标对比
- 遮挡影响分析报告
- 可视化图表和统计信息

## 注意事项

1. **数据路径**：确保KITTI数据集路径正确，图像和标注文件对应
2. **检测质量**：跟踪性能很大程度上依赖于检测质量，建议使用高质量的检测器
3. **计算资源**：FairMOT需要较多GPU资源，DeepSORT次之，SORT最轻量
4. **序列选择**：不同序列的难度不同，建议对所有序列进行评估以获得全面结论

## 文件说明

- `gen_det_yolov8.py`: 使用YOLOv8生成检测结果
- `utils.py`: 工具函数，包括KITTI数据加载等
- `scripts/run_*.py`: 各算法的运行脚本
- `analysis/eval_*.py`: 评估和分析脚本
- `src/*/`: 各算法的实现源码

## 贡献

欢迎提交Issue和Pull Request来改进本项目。

## 许可证

本项目仅供学术研究使用。

## 参考文献

- SORT: Simple Online and Realtime Tracking
- DeepSORT: Simple Online and Realtime Tracking with a Deep Association Metric
- FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking
- KITTI Dataset: The KITTI Vision Benchmark Suite

## 联系方式

如有问题或建议，请通过yinyouaiai@163.com联系。

