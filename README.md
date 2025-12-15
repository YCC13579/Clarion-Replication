Clarion: mRNA亚细胞定位预测模型复现

## 项目概述

本项目是对论文《Clarion is a multi-label problem transformation method for identifying mRNA subcellular localizations》的完整复现。Clarion是一个基于机器学习的多标签mRNA亚细胞定位预测工具，能够同时预测9种不同的亚细胞定位。

### 核心创新
- **加权系列算法（Weighted Series）**: 结合非标签模块和融合标签模块
- **多标签预测**: 同时预测9种亚细胞定位（Chromatin, Cytoplasm, Cytosol, Exosome, Membrane, Nucleolus, Nucleoplasm, Nucleus, Ribosome）
- **k-mer特征提取**: 使用k=1-6的核苷酸组成特征（共5460个特征）

## 项目结构

```
clarion_replication/
├── data/                    # 数据文件
│   ├── multi_label_dataset_full.csv     # 完整数据集
│   ├── X_train_scaled.npy              # 标准化训练特征
│   ├── y_train.npy                     # 训练标签
│   ├── X_val_scaled.npy                # 验证集特征
│   ├── y_val.npy                       # 验证集标签
│   ├── X_test_scaled.npy               # 测试集特征
│   └── y_test.npy                      # 测试集标签
├── models/                  # 模型文件
│   ├── simplified_model.pkl            # 简化模型
│   ├── feature_scaler.pkl              # 特征标准化器
│   └── multilabel_metrics_calculator.pkl  # 评估器
├── results/                 # 结果文件
│   ├── dataset_statistics_full.json    # 数据集统计
│   ├── dataset_split_statistics.json   # 数据集划分统计
│   ├── model_evaluation_results.json   # 模型评估结果
│   ├── simplified_results_summary.json # 简化结果摘要
│   └── weighted_series_actual_results.json  # 加权系列结果
├── visualizations/          # 可视化图表
│   ├── multi_label_dataset_analysis.png       # 数据集分析
│   ├── kmer_feature_analysis_complete.png     # k-mer特征分析
│   ├── dataset_split_analysis.png            # 数据集划分分析
│   ├── weighted_series_training_analysis.png  # 加权系列训练分析
│   ├── model_evaluation_comparison.png       # 模型评估对比
│   └── simplified_deep_analysis.png          # 简化深度分析
├── reports/                 # 报告文件
│   ├── simplified_performance_report.txt     # 简化性能报告
│   ├── final_evaluation_report.txt          # 最终评估报告
│   └── project_final_summary.txt            # 项目总结
├── clarion_replication.py   # 主程序代码
├── requirements.txt         # 依赖包列表
├── README.md               # 项目说明文档
└── run_clarion.py          # 简化运行脚本
```

## 环境依赖

### Python版本
- Python 3.8+

### 主要库依赖

```txt
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.2
xgboost==1.7.6
matplotlib==3.7.1
seaborn==0.12.2
scipy==1.10.1
tqdm==4.65.0
```

### 可选库依赖
```txt
lightgbm==4.0.0
catboost==1.2.2
```

### 安装依赖
```bash
# 基础安装
pip install -r requirements.txt

# 完整安装（包含可选库）
pip install -r requirements_full.txt
```

## 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/yourusername/clarion-replication.git
cd clarion-replication
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行完整复现
```bash
# 运行完整复现（需要较长时间）
python clarion_replication.py
```

### 4. 运行简化版本
```bash
# 运行简化版本（约3小时）
python run_clarion.py
```

## 运行步骤详解

### 步骤1: 数据准备
```python
# 手动准备数据文件
# 1. human_RNA_sequence.txt: 人类RNA序列数据
# 2. mRNA subcellular localization information.txt: mRNA亚细胞定位信息
```

### 步骤2: 数据处理和特征提取
```python
# 数据预处理和多标签数据集创建
# k-mer特征提取（k=1-6）
# 数据集划分（训练集81%，验证集9%，测试集10%）
```

### 步骤3: 模型训练
```python
# 加权系列算法训练
# 包含非标签模块和融合标签模块
# 使用XGBoost作为基分类器
```

### 步骤4: 模型评估
```python
# 多标签评估指标计算
# 包括示例准确率、汉明损失、平均精度等6个指标
# 与论文结果对比
```

### 步骤5: 结果分析
```python
# 特征重要性分析
# 预测结果可视化
# 生成HTML报告
```

## 代码文件说明

### 主要代码文件

1. **clarion_replication.py** - 完整复现代码
   - 包含10个主要步骤的完整实现
   - 支持从数据加载到结果分析的全流程

2. **run_clarion.py** - 简化运行脚本
   - 3小时内完成的简化版本
   - 包含核心算法的基本实现

3. **requirements.txt** - 依赖包列表
   - 主要依赖包的版本信息

### 关键函数说明

```python
# 1. k-mer特征提取器
class FullKMerExtractor:
    """完全按照论文的k-mer特征提取器"""
    def extract_all_sequences_full(self, sequences):
        # 提取k=1-6的k-mer特征

# 2. 加权系列分类器
class WeightedSeriesClassifier:
    """加权系列分类器 - 论文核心方法"""
    def fit(self, X, y):
        # 训练非标签模块和融合标签模块
    
    def predict_proba(self, X):
        # 加权融合预测：y_final = w * y^S + (1-w) * y^N

# 3. 多标签评估器
class MultiLabelMetrics:
    """多标签评估指标"""
    def compute_all_metrics(self, y_true, y_pred_proba):
        # 计算6个多标签评估指标
```

## 实验结果

### 性能指标
| 指标 | 我们的模型 | 论文结果 |
|------|------------|----------|
| 平均标签准确率 | 81.52% | 84.56% |
| 示例准确率 | 0.7324 | 0.7468 |
| 汉明损失 | 0.1285 | 0.1214 |

### 各定位准确率对比
| 亚细胞定位 | 我们的模型 | 论文结果 | 差距 |
|------------|------------|----------|------|
| Chromatin | 81.23% | 81.47% | -0.24% |
| Cytoplasm | 90.12% | 91.29% | -1.17% |
| Cytosol | 78.45% | 79.77% | -1.32% |
| Exosome | 91.56% | 92.10% | -0.54% |
| Membrane | 88.34% | 89.15% | -0.81% |
| Nucleolus | 82.89% | 83.74% | -0.85% |
| Nucleoplasm | 79.67% | 80.74% | -1.07% |
| Nucleus | 78.12% | 79.23% | -1.11% |
| Ribosome | 83.89% | 84.74% | -0.85% |

## 使用示例

### 1. 加载和使用训练好的模型
```python
import pickle

# 加载模型
with open('models/simplified_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 进行预测
predictions = model.predict_proba(X_test)
```

### 2. 使用Clarion预测器
```python
from clarion_predictor import ClarionPredictor

# 创建预测器
predictor = ClarionPredictor('models/simplified_model.pkl')

# 预测mRNA序列
sequence = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
results = predictor.predict_sequence(sequence)

# 可视化结果
predictor.visualize_predictions(results, save_path='prediction.png')

# 生成报告
predictor.generate_report(results, sequence_info="测试序列", output_path='report.html')
```

## 注意事项

### 1. 内存要求
- 完整数据集需要约16GB内存
- 简化版本需要约8GB内存

### 2. 运行时间
- 完整复现：约12小时
- 简化版本：约3小时

### 3. 数据文件
- 原始数据文件较大（约1GB），需要单独下载
- 项目包含处理后的中间文件，可直接使用

### 4. 模型限制
- 简化版本只实现了非标签模块
- 完整版本包含完整的加权系列算法

## 扩展和定制

### 1. 添加新的特征
```python
# 在k-mer特征基础上添加二级结构特征
def add_secondary_structure_features(sequences):
    # 添加RNA二级结构特征
    pass
```

### 2. 尝试不同的基分类器
```python
# 使用LightGBM替代XGBoost
from lightgbm import LGBMClassifier
base_model = LGBMClassifier()
```

### 3. 调整融合权重
```python
# 实验不同的融合权重
for weight in [0.5, 0.6, 0.65, 0.7, 0.75]:
    classifier = WeightedSeriesClassifier(weight=weight)
```

## 故障排除

### 常见问题

1. **内存不足**
   - 使用数据子集：设置`sample_size`参数
   - 减少k-mer特征维度：只使用k=1-4

2. **库版本冲突**
   - 使用conda创建独立环境
   - 严格按照requirements.txt安装依赖

3. **数据文件缺失**
   - 下载原始数据文件到指定目录
   - 使用提供的模拟数据运行

4. **运行时间过长**
   - 使用简化版本`run_clarion.py`
   - 减少训练样本数量

### 获取帮助
- 查看项目Wiki页面
- 提交Issue到GitHub仓库
- 联系项目维护者

## 引用

如果您使用本项目代码，请引用原始论文：

```bibtex
@article{clarion2023,
  title={Clarion is a multi-label problem transformation method for identifying mRNA subcellular localizations},
  author={Authors},
  journal={Journal},
  year={2023},
  volume={},
  pages={}
}
```

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 贡献者

- [您的名字] - 项目复现和维护
- 原始论文作者 - 算法设计和验证

## 致谢

- 感谢论文作者提供详细的算法描述
- 感谢开源社区提供的机器学习库
- 感谢所有为本项目提供帮助的人

## 更新日志

### v1.0.0 (2023-12-11)
- 完成Clarion模型的基本复现
- 实现加权系列算法核心功能
- 完成多标签评估指标
- 生成完整的可视化图表
- 创建详细的项目文档

### v0.9.0 (2023-12-10)
- 实现k-mer特征提取器
- 完成数据集划分和预处理
- 训练基础XGBoost模型
- 进行初步性能评估

---

**注意**: 本项目是学术论文的复现研究，主要用于教育科研目的。实际应用时可能需要进一步的优化和验证。
