import os
import json
import torch
import logging
import warnings
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm

# 过滤掉特定的警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*cuda.*")
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint.*")
warnings.filterwarnings("ignore", message=".*has generative capabilities.*")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinanceDataset(Dataset):
    def __init__(
        self, texts: List[str], tokenizer: BertTokenizer, max_length: int = 256
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 预处理所有文本
        self.encodings = [self._encode_text(text) for text in texts]

    def _encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        # 编码并转换为张量
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        # 转换为一维张量
        return {
            "input_ids": inputs["input_ids"][0],  # 直接获取第一个维度
            "attention_mask": inputs["attention_mask"][0],  # 直接获取第一个维度
        }

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.encodings[idx]


class FinanceModel:
    def __init__(
        self,
        model_dir: str = "models",
        model_name: str = "bert-base-chinese",
        max_length: int = 256,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        accumulation_steps: int = 2,
    ):
        self.model_dir = model_dir
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.accumulation_steps = accumulation_steps

        # 确保模型目录存在
        os.makedirs(model_dir, exist_ok=True)

        # 初始化设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")

        # 初始化模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)

        # 添加领域特殊标记
        special_tokens = {
            "additional_special_tokens": [
                "[STOCK]",
                "[TERM]",
                "[INDICATOR]",
            ],
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # 移动模型到指定设备
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
            self.model.to(self.device)

    def prepare_data(self, processed_data_path: str) -> List[str]:
        """准备训练数据"""
        with open(processed_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        processed_texts = []
        for item in data:
            # 提取金融实体
            finance_entities = item["features"]["finance_entities"]
            stocks = " ".join(
                [f"[STOCK]{s}[/STOCK]" for s in finance_entities["stocks"]]
            )
            terms = " ".join([f"[TERM]{t}[/TERM]" for t in finance_entities["terms"]])
            indicators = " ".join(
                [f"[INDICATOR]{i}[/INDICATOR]" for i in finance_entities["indicators"]]
            )

            # 构建训练文本
            text = f"{item['cleaned_text']} {stocks} {terms} {indicators}"
            processed_texts.append(text)

        return processed_texts

    def train(self, texts: List[str], num_epochs: int = 3) -> Dict[str, float]:
        """训练模型"""
        # 准备数据集
        dataset = FinanceDataset(texts, self.tokenizer, self.max_length)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

        # 优化器和学习率调度器
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps,
        )

        # 混合精度训练
        scaler = GradScaler()

        # TensorBoard
        writer = SummaryWriter(os.path.join(self.model_dir, "runs"))

        # 训练循环
        global_step = 0
        total_loss = 0.0
        best_loss = float("inf")

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for step, batch in enumerate(progress_bar):
                # 将数据移到GPU
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # 随机遮蔽一些token用于训练
                masked_indices = torch.rand(input_ids.shape, device=self.device) < 0.15
                masked_indices[input_ids == self.tokenizer.cls_token_id] = False
                masked_indices[input_ids == self.tokenizer.sep_token_id] = False
                masked_indices[input_ids == self.tokenizer.pad_token_id] = False

                labels = input_ids.clone()
                labels[~masked_indices] = -100  # 只计算被遮蔽token的损失

                masked_input_ids = input_ids.clone()
                masked_input_ids[masked_indices] = self.tokenizer.mask_token_id

                # 清除梯度
                optimizer.zero_grad()

                # 混合精度训练
                with autocast():
                    outputs = self.model(
                        input_ids=masked_input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / self.accumulation_steps

                # 反向传播
                scaler.scale(loss).backward()

                if (step + 1) % self.accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    # 参数更新
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                # 记录损失
                total_loss += loss.item() * self.accumulation_steps
                epoch_loss += loss.item() * self.accumulation_steps
                global_step += 1

                # 更新进度条
                progress_bar.set_postfix(
                    {
                        "loss": loss.item() * self.accumulation_steps,
                        "avg_loss": total_loss / global_step,
                    }
                )

                # 记录到TensorBoard
                if global_step % 100 == 0:
                    writer.add_scalar(
                        "Loss/train", total_loss / global_step, global_step
                    )

            # 计算epoch平均损失
            avg_epoch_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1}: Average Loss = {avg_epoch_loss:.4f}")

            # 保存最佳模型
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                self.save_model("best_model")

            # 定期保存检查点
            if (epoch + 1) % 1 == 0:
                self.save_model(f"checkpoint-epoch-{epoch + 1}")

        writer.close()
        return {"final_loss": total_loss / global_step, "best_loss": best_loss}

    def save_model(self, name: str = "best_model") -> None:
        """保存模型"""
        save_dir = os.path.join(self.model_dir, name)
        os.makedirs(save_dir, exist_ok=True)

        # 如果使用了DataParallel，需要保存原始模型
        model_to_save = (
            self.model.module
            if isinstance(self.model, torch.nn.DataParallel)
            else self.model
        )
        model_to_save.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        # 保存配置信息
        config = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        }

        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def load_model(self, name: str = "best_model") -> None:
        """加载模型"""
        load_dir = os.path.join(self.model_dir, name)
        self.model = BertForMaskedLM.from_pretrained(load_dir)
        self.tokenizer = BertTokenizer.from_pretrained(load_dir)

        # 移动模型到设备并根据需要使用DataParallel
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
            self.model.to(self.device)

    def fill_mask(self, text: str, top_k: int = 5) -> List[Dict[str, float]]:
        """完成带有[MASK]的文本"""
        self.model.eval()

        # 处理输入
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        # 将输入移到GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 预测
        with torch.no_grad():
            model_to_use = (
                self.model.module
                if isinstance(self.model, torch.nn.DataParallel)
                else self.model
            )
            outputs = model_to_use(**inputs)
            predictions = outputs.logits

        # 找到[MASK]的位置
        mask_token_index = torch.where(
            inputs["input_ids"] == self.tokenizer.mask_token_id
        )[1]

        if len(mask_token_index) == 0:
            return []

        # 获取[MASK]位置的预测结果
        mask_token_logits = predictions[0, mask_token_index, :]

        # 获取top k个预测结果
        top_k_values, top_k_indices = torch.topk(
            torch.softmax(mask_token_logits, dim=-1), top_k
        )

        return [
            {
                "token": self.tokenizer.decode([token_id.item()]),
                "score": score.item(),
            }
            for token_id, score in zip(top_k_indices[0], top_k_values[0])
        ]


def main():
    """训练模型"""
    model = FinanceModel()

    try:
        # 准备数据
        texts = model.prepare_data("data/processed/processed_data.json")
        logger.info(f"加载了{len(texts)}条训练数据")

        # 训练模型
        results = model.train(texts)
        logger.info("\n训练完成")
        logger.info(f"最终损失: {results['final_loss']:.4f}")
        logger.info(f"最佳损失: {results['best_loss']:.4f}")

        # 测试遮蔽语言模型
        test_texts = [
            "今日[MASK]股市场整体表现良好",  # 预期：A
            "央行下调[MASK]率以提振经济",  # 预期：利
            "[MASK]指数大幅上涨",  # 预期：沪/深/创业板
            "投资者关注[MASK]公司财报",  # 预期：科技/银行
            "本月[MASK]数据超出市场预期",  # 预期：GDP/CPI
        ]

        for text in test_texts:
            predictions = model.fill_mask(text)
            logger.info(f"\n遮蔽预测示例:\n输入: {text}")
            for pred in predictions:
                logger.info(f"预测: {pred['token']}, 概率: {pred['score']:.4f}")

    except FileNotFoundError:
        logger.error("请先运行preprocessor.py处理数据")


if __name__ == "__main__":
    main()
