import os
import json
import numpy as np
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib


class TextClassifier:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), min_df=2
        )
        self.label_encoder = LabelEncoder()
        self.classifier = LinearSVC(C=1.0, class_weight="balanced", max_iter=1000)

        # 确保模型目录存在
        os.makedirs(model_dir, exist_ok=True)

    def prepare_data(self, processed_data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        # 加载预处理后的数据
        with open(processed_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 提取文本和特征
        texts = []
        features = []
        labels = []

        for item in data:
            # 合并清理后的文本和分词结果
            text = f"{item['cleaned_text']} {' '.join(item['segmented_words'])}"
            texts.append(text)

            # 提取数值特征
            feature = [
                item["features"]["sentiment_score"],
                item["features"]["text_length"],
                item["features"]["word_count"],
                item["features"]["contains_url"],
                item["features"]["contains_mention"],
                item["features"]["contains_topic"],
            ]
            features.append(feature)

            # 使用主题作为标签（示例，实际应用中需要真实标签）
            if item["topics"]:
                labels.append(item["topics"][0][0])  # 使用最主要的主题词作为标签
            else:
                labels.append("其他")

        # 转换特征
        X_text = self.vectorizer.fit_transform(texts)
        X_features = np.array(features)

        # 合并文本特征和数值特征
        X = np.hstack((X_text.toarray(), X_features))

        # 转换标签
        y = self.label_encoder.fit_transform(labels)

        return X, y

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """训练模型"""
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 训练模型
        self.classifier.fit(X_train, y_train)

        # 评估模型
        y_pred = self.classifier.predict(X_test)

        # 保存评估报告
        report = classification_report(
            y_test, y_pred, target_names=self.label_encoder.classes_, output_dict=True
        )

        # 保存混淆矩阵
        cm = confusion_matrix(y_test, y_pred)

        # 保存评估结果
        results = {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "accuracy": report["accuracy"],
            "macro_avg_f1": report["macro avg"]["f1-score"],
        }

        with open(
            os.path.join(self.model_dir, "evaluation.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return results

    def save_model(self):
        """保存模型和相关组件"""
        # 保存TF-IDF向量器
        joblib.dump(self.vectorizer, os.path.join(self.model_dir, "vectorizer.joblib"))

        # 保存标签编码器
        joblib.dump(
            self.label_encoder, os.path.join(self.model_dir, "label_encoder.joblib")
        )

        # 保存分类器
        joblib.dump(self.classifier, os.path.join(self.model_dir, "classifier.joblib"))

    def load_model(self):
        """加载已训练的模型和组件"""
        self.vectorizer = joblib.load(os.path.join(self.model_dir, "vectorizer.joblib"))
        self.label_encoder = joblib.load(
            os.path.join(self.model_dir, "label_encoder.joblib")
        )
        self.classifier = joblib.load(os.path.join(self.model_dir, "classifier.joblib"))

    def predict(self, texts: List[str], raw_features: List[Dict]) -> List[str]:
        """对新文本进行预测"""
        # 转换文本特征
        X_text = self.vectorizer.transform(texts)

        # 提取数值特征
        features = []
        for feat in raw_features:
            feature = [
                feat["sentiment_score"],
                feat["text_length"],
                feat["word_count"],
                feat["contains_url"],
                feat["contains_mention"],
                feat["contains_topic"],
            ]
            features.append(feature)

        X_features = np.array(features)

        # 合并特征
        X = np.hstack((X_text.toarray(), X_features))

        # 预测
        y_pred = self.classifier.predict(X)

        # 转换回标签
        return self.label_encoder.inverse_transform(y_pred)


def main():
    """训练模型"""
    classifier = TextClassifier()

    try:
        # 准备数据
        X, y = classifier.prepare_data("data/processed/processed_data.json")

        # 训练模型
        results = classifier.train(X, y)
        print("\n模型评估结果：")
        print(f"准确率: {results['accuracy']:.4f}")
        print(f"宏平均F1: {results['macro_avg_f1']:.4f}")

        # 保存模型
        classifier.save_model()
        print("\n模型已保存到models目录")

    except FileNotFoundError:
        print("请先运行preprocessor.py处理数据")


if __name__ == "__main__":
    main()
