import re
import json
import jieba
import pandas as pd
from typing import List, Dict, Any
from snownlp import SnowNLP


class TextPreprocessor:
    def __init__(self):
        # 加载停用词
        self.stopwords = set(
            [
                "的",
                "了",
                "在",
                "是",
                "我",
                "有",
                "和",
                "就",
                "不",
                "人",
                "都",
                "一",
                "一个",
                "上",
                "也",
                "很",
                "到",
                "说",
                "要",
                "去",
                "你",
                "会",
                "着",
                "没有",
                "看",
                "好",
                "自己",
                "这",
            ]
        )

    def clean_text(self, text: str) -> str:
        """清理文本"""
        # 去除HTML标签
        text = re.sub(r"<[^>]+>", "", text)
        # 去除URL
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )
        # 去除@用户
        text = re.sub(r"@[\w\-]+", "", text)
        # 去除话题标签
        text = re.sub(r"#[\w\-]+#", "", text)
        # 去除表情符号
        text = re.sub(r"\[[\w\-\/]+\]", "", text)
        return text.strip()

    def segment_text(self, text: str) -> List[str]:
        """分词"""
        words = jieba.lcut(text)
        # 去除停用词
        words = [w for w in words if w not in self.stopwords]
        return words

    def extract_features(self, text: str) -> Dict[str, Any]:
        """特征提取"""
        s = SnowNLP(text)
        return {
            "sentiment_score": s.sentiments,  # 情感得分
            "keywords": s.keywords(3),  # 关键词
            "text_length": len(text),  # 文本长度
            "word_count": len(self.segment_text(text)),  # 词数
            "contains_url": 1 if "http" in text else 0,  # 是否包含URL
            "contains_mention": 1 if "@" in text else 0,  # 是否包含@
            "contains_topic": 1 if "#" in text else 0,  # 是否包含话题
        }

    def process_weibo_data(self, input_file: str, output_file: str):
        """处理微博数据"""
        # 读取JSON数据
        with open(input_file, "r", encoding="utf-8") as f:
            posts = json.load(f)

        processed_posts = []
        for post in posts:
            # 清理文本
            cleaned_text = self.clean_text(post["text"])
            # 分词
            words = self.segment_text(cleaned_text)
            # 提取特征
            features = self.extract_features(cleaned_text)

            processed_post = {
                "original_text": post["text"],
                "cleaned_text": cleaned_text,
                "segmented_words": words,
                "features": features,
                "topics": post.get("topics", []),
                "finance_features": post.get("finance_features", {}),
                "metadata": {
                    "user": post["user"],
                    "created_at": post["created_at"],
                    "reposts": post["reposts"],
                    "comments": post["comments"],
                },
            }
            processed_posts.append(processed_post)

        # 保存处理后的数据
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_posts, f, ensure_ascii=False, indent=2)

        # 转换为DataFrame方便后续分析
        df = pd.DataFrame(processed_posts)
        df.to_csv(output_file.replace(".json", ".csv"), index=False, encoding="utf-8")

        print(f"数据预处理完成，共处理{len(processed_posts)}条记录")
        return df


def main():
    """测试预处理功能"""
    preprocessor = TextPreprocessor()

    try:
        df = preprocessor.process_weibo_data(
            "data/raw/weibo_data.json", "data/processed/processed_data.json"
        )
        print("数据预处理样例：")
        print(df.head())

    except FileNotFoundError:
        print("请先运行scraper.py采集数据")


if __name__ == "__main__":
    main()
