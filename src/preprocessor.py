import re
import json
import jieba
import jieba.posseg as pseg
import pandas as pd
from typing import List, Dict, Any, Tuple
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

        # 加载金融领域词典
        self.load_finance_lexicons()

    def load_finance_lexicons(self):
        """加载金融领域词典"""
        self.finance_terms = {}

        # 加载股票词典
        with open("lexicons/finance_stocks.tsv", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    term, weight = parts[0], parts[1]
                else:
                    term, weight = parts[0], "1.0"  # 如果没有权重，默认使用1.0
                self.finance_terms[term] = float(weight)
                jieba.add_word(term, freq=None, tag="STOCK")

        # 加载金融术语词典
        with open("lexicons/finance_terms.tsv", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    term, weight = parts[0], parts[1]
                else:
                    term, weight = parts[0], "1.0"  # 如果没有权重，默认使用1.0
                self.finance_terms[term] = float(weight)
                jieba.add_word(term, freq=None, tag="TERM")

        # 加载金融指标词典
        with open("lexicons/finance_indicators.tsv", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    term, weight = parts[0], parts[1]
                else:
                    term, weight = parts[0], "1.0"  # 如果没有权重，默认使用1.0
                self.finance_terms[term] = float(weight)
                jieba.add_word(term, freq=None, tag="INDICATOR")

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

    def segment_text(self, text: str) -> List[Tuple[str, str]]:
        """分词并标注词性"""
        words = []
        # 使用posseg进行词性标注
        for word, flag in pseg.cut(text):
            # 去除停用词，但保留金融术语
            if word not in self.stopwords or word in self.finance_terms:
                words.append((word, flag))
        return words

    def extract_features(self, text: str) -> Dict[str, Any]:
        """特征提取"""
        s = SnowNLP(text)
        word_pairs = self.segment_text(text)
        words = [word for word, _ in word_pairs]

        # 提取金融实体
        finance_entities = {
            "stocks": [
                word
                for word, flag in word_pairs
                if word in self.finance_terms and flag == "n"
            ],
            "terms": [
                word
                for word, flag in word_pairs
                if word in self.finance_terms and flag == "n"
            ],
            "indicators": [
                word
                for word, flag in word_pairs
                if word in self.finance_terms and flag == "n"
            ],
        }

        return {
            "sentiment_score": s.sentiments,  # 情感得分
            "keywords": s.keywords(3),  # 关键词
            "text_length": len(text),  # 文本长度
            "word_count": len(words),  # 词数
            "contains_url": 1 if "http" in text else 0,  # 是否包含URL
            "contains_mention": 1 if "@" in text else 0,  # 是否包含@
            "contains_topic": 1 if "#" in text else 0,  # 是否包含话题
            "finance_entities": finance_entities,  # 金融实体
            "finance_entity_count": sum(
                len(entities) for entities in finance_entities.values()
            ),  # 金融实体数量
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
            word_pairs = self.segment_text(cleaned_text)
            words = [word for word, _ in word_pairs]
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
