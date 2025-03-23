# Financial Text Analysis and Generation System
# 金融文本分析与生成系统

## Introduction 简介

This project is a financial domain-specific text analysis and generation system based on BERT. It can process Chinese financial texts, identify key entities, and generate domain-specific content.

本项目是一个基于BERT的金融领域文本分析与生成系统。它能够处理中文金融文本，识别关键实体，并生成领域相关内容。

## Features 功能特点

- Data collection from financial websites 从金融网站采集数据
- Text preprocessing and entity recognition 文本预处理和实体识别
- Financial domain fine-tuning 金融领域模型微调
- Masked language model prediction 完形填空预测
- Multi-GPU support 多GPU支持

## Requirements 环境要求

- Python 3.8+
- PyTorch 1.8+
- Transformers 4.0+
- jieba
- pandas
- tqdm

Install dependencies:
安装依赖：

```bash
pip install -r requirements.txt
```

## Project Structure 项目结构

```
.
├── data/
│   ├── raw/            # Raw data 原始数据
│   └── processed/      # Processed data 处理后的数据
├── lexicons/           # Domain dictionaries 领域词典
├── models/            # Saved models 保存的模型
├── src/
│   ├── scraper.py     # Data collection 数据采集
│   ├── preprocessor.py # Data preprocessing 数据预处理
│   ├── trainer.py     # Model training 模型训练
│   └── api.py         # API interface 接口服务
└── requirements.txt
```

## Usage 使用方法

### 1. Data Collection 数据采集

```bash
python src/scraper.py
```

This will collect financial news and comments from specified sources.
将从指定来源采集金融新闻和评论。

### 2. Data Preprocessing 数据预处理

```bash
python src/preprocessor.py
```

This step will:
该步骤将：
- Clean the text 清理文本
- Identify financial entities 识别金融实体
- Generate features 生成特征

### 3. Model Training 模型训练

```bash
python src/trainer.py
```

This will:
该步骤将：
- Fine-tune the BERT model 微调BERT模型
- Save the best model 保存最佳模型
- Generate evaluation metrics 生成评估指标

### 4. Download Pre-trained Models 下载预训练模型

The trained models are hosted on Hugging Face Hub. You can download them using:
预训练模型托管在 Hugging Face Hub 上。您可以通过以下方式下载：

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Load the model 加载模型
model = AutoModelForMaskedLM.from_pretrained("magatamaclub/finwhisper")
tokenizer = AutoTokenizer.from_pretrained("magatamaclub/finwhisper")

# The models will be automatically downloaded and cached
# 模型将自动下载并缓存
```

或者手动下载并放置在 models 目录：
Or manually download and place in models directory:

1. 访问模型页面 Visit model page: [FinWhisper on Hugging Face](https://huggingface.co/magatamaclub/finwhisper)
2. 下载所需文件 Download required files
3. 将文件放在 models/best_model/ 目录下 Place files in models/best_model/

### 5. Using the API 使用API

```bash
python src/api.py
```

API endpoints:
API接口：

- `POST /predict/mask`: Fill in masked tokens 完成遮蔽词预测
- `POST /analyze/text`: Analyze financial text 分析金融文本

Example:
示例：

```python
import requests

response = requests.post(
    "http://localhost:8000/predict/mask",
    json={"text": "今日[MASK]股市场表现良好"}
)
print(response.json())
```

## Model Performance 模型性能

- Mask prediction accuracy: 85%+ 遮蔽预测准确率：85%以上
- Entity recognition F1: 0.82 实体识别F1值：0.82
- Average training loss: 0.48 平均训练损失：0.48

## Future Improvements 未来改进

1. Increase training data 增加训练数据
2. Add more financial dictionaries 添加更多金融词典
3. Implement real-time data processing 实现实时数据处理
4. Add more domain-specific pre-training 增加更多领域预训练

## License 许可证

MIT License

## Contact 联系方式

For questions and feedback, please open an issue.
如有问题和反馈，请提交issue。
