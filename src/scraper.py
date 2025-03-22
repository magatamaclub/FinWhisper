import os
import json
import random
import time
import requests
import jieba
from collections import defaultdict
import jieba.analyse as analyse


class FinanceLexiconLoader:
    def __init__(self):
        self.lexicons = {"stocks": {}, "terms": {}, "indicators": {}}
        self._init_jieba()

    def _init_jieba(self):
        """初始化金融词典"""
        for category in ["stocks", "terms", "indicators"]:
            path = os.path.join("lexicons", f"finance_{category}.tsv")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        text = line.strip()
                        # 从右向左查找第一个非数字和非点的位置
                        i = len(text) - 1
                        while i >= 0 and (text[i].isdigit() or text[i] == "."):
                            i -= 1
                        # 分割词语和权重
                        word = text[: i + 1].strip()
                        weight = float(text[i + 1 :])
                        self.lexicons[category][word] = weight
                        if category in ["stocks", "terms"]:
                            jieba.add_word(word, freq=2000, tag="nz")

    def enhance_analysis(self, text):
        """金融特征分析"""
        return {
            "stock_mentions": [
                word for word in self.lexicons["stocks"] if word in text
            ],
            "financial_terms": [
                word for word in self.lexicons["terms"] if word in text
            ],
            "economic_indicators": [
                word for word in self.lexicons["indicators"] if word in text
            ],
        }


class TopicExtractor:
    def __init__(self):
        self.stopwords = set(["转发微博", "via", "网页链接", "//", "@"])

    def extract(self, text, methods=["tfidf", "textrank"]):
        """多算法融合主题提取"""
        topics = []
        if "tfidf" in methods:
            topics += analyse.extract_tags(text, topK=3, withWeight=True)
        if "textrank" in methods:
            topics += analyse.textrank(text, topK=3, withWeight=True)

        merged = defaultdict(float)
        for word, weight in topics:
            if word not in self.stopwords:
                merged[word] += weight

        return sorted(merged.items(), key=lambda x: x[1], reverse=True)[:5]


class WeiboScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "X-Requested-With": "XMLHttpRequest",
            "MWeibo-Pwa": "1",
            "Referer": "https://m.weibo.cn/search?containerid=100103type%3D1%26q%3D",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Dest": "empty",
            "Cookie": "_T_WM=67649381483; SCF=AjsEiVvjv1KWGGwxXWjxB9RjUpwjwLk3nFZ7wbhC7bqXKrGarLr4bn7saXXyRPsJ6hCiSw.;",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.topic_extractor = TopicExtractor()
        self.finance_lexicon = FinanceLexiconLoader()

    def scrape_posts(self, keyword, max_pages=10, max_retries=3):
        """采集微博文章"""
        all_posts = []
        seen_ids = set()  # 用于快速查找重复
        stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "total_cards": 0,
            "valid_posts": 0,
            "duplicates": 0,
            "errors": 0,
            "retries": 0,
            "start_time": time.time(),
        }
        base_url = "https://m.weibo.cn/api/container/getIndex"

        print(f"\n开始采集关键词: {keyword}")
        print(f"计划采集页数: {max_pages}")
        print("=" * 50)

        for page in range(1, max_pages + 1):
            retry_count = 0
            success = False

            while retry_count < max_retries and not success:
                try:
                    params = {
                        "containerid": f"100103type=1&q={keyword}",
                        "page": page,
                        "count": 20,
                        "extparam": "mix_type=extended",
                        "current_page": page,
                        "since_id": "",
                        "page_type": "searchall",
                    }

                    print(f"\n正在请求第{page}页")
                    print(f"URL: {base_url}")
                    print(f"参数: {params}")

                    stats["total_requests"] += 1
                    response = self.session.get(base_url, params=params, timeout=10)
                    print(f"响应状态码: {response.status_code}")

                    if response.status_code != 200:
                        raise Exception(f"请求失败: HTTP {response.status_code}")

                    data = response.json()
                    stats["successful_requests"] += 1
                    print(
                        f"数据结构:\n{json.dumps(data, ensure_ascii=False, indent=2)[:500]}..."
                    )

                    if data.get("ok") != 1:
                        raise Exception(f"接口返回错误: {data.get('msg', '未知错误')}")

                    if "data" not in data or "cards" not in data["data"]:
                        raise Exception("数据结构异常：没有找到cards数据")

                    cards = data["data"]["cards"]
                    stats["total_cards"] += len(cards)
                    print(f"\n本页包含 {len(cards)} 条卡片数据")

                    for card in cards:
                        try:
                            # 检查是否为有效的微博卡片
                            if not isinstance(card, dict):
                                continue

                            print(f"卡片类型: {card.get('card_type')}")

                            # 检查卡片类型并获取微博内容
                            if "mblog" not in card:
                                if "card_group" in card:
                                    for item in card["card_group"]:
                                        if isinstance(item, dict) and "mblog" in item:
                                            mblog = item["mblog"]
                                            break
                                    else:
                                        continue
                                else:
                                    continue
                            else:
                                mblog = card["mblog"]

                            if not isinstance(mblog, dict):
                                continue

                            # 清理HTML标签和特殊字符
                            text = mblog["text"]
                            text = text.replace("<br />", "\n")
                            text = text.replace("&quot;", '"')
                            text = " ".join(
                                [
                                    t
                                    for t in text.split()
                                    if not t.startswith(("<", "http"))
                                ]
                            )

                            # 提取话题和分析
                            topics = self.topic_extractor.extract(text)
                            finance_features = self.finance_lexicon.enhance_analysis(
                                text
                            )

                            # 检查重复
                            post_id = mblog.get("id", "")
                            if post_id in seen_ids:
                                print(f"跳过重复微博: {post_id}")
                                stats["duplicates"] += 1
                                continue
                            seen_ids.add(post_id)

                            # 提取用户信息
                            user_info = mblog.get("user", {})

                            # 组装数据
                            post = {
                                "id": post_id,
                                "bid": mblog.get("bid", ""),
                                "created_at": mblog.get("created_at", ""),
                                "text": text,
                                "text_length": len(text),
                                "topics": topics,
                                "finance_features": finance_features,
                                "reposts": mblog.get("reposts_count", 0),
                                "comments": mblog.get("comments_count", 0),
                                "attitudes": mblog.get("attitudes_count", 0),
                                "source": mblog.get("source", ""),
                                "user": {
                                    "id": user_info.get("id", ""),
                                    "screen_name": user_info.get(
                                        "screen_name", "未知用户"
                                    ),
                                    "followers": user_info.get("followers_count", 0),
                                    "following": user_info.get("follow_count", 0),
                                    "verified": user_info.get("verified", False),
                                    "verified_type": user_info.get("verified_type", -1),
                                    "description": user_info.get("description", ""),
                                    "gender": user_info.get("gender", ""),
                                },
                                "isLongText": mblog.get("isLongText", False),
                                "page_info": mblog.get("page_info", {}),
                                "visible": mblog.get("visible", {}).get("type", 0),
                            }
                            all_posts.append(post)
                            stats["valid_posts"] += 1
                            print(f"成功解析微博: {post['id']}")

                        except Exception as e:
                            print(f"解析微博失败: {str(e)}")
                            stats["errors"] += 1

                    print(f"\n第{page}页采集完成，当前共有{len(all_posts)}条数据")
                    success = True

                    # 动态调整延迟
                    delay = 3 + random.random() * 2
                    if len(cards) > 20:
                        delay += 2
                    print(f"等待 {delay:.1f} 秒后继续...")
                    time.sleep(delay)

                except Exception as e:
                    retry_count += 1
                    stats["errors"] += 1
                    stats["retries"] += 1
                    if retry_count < max_retries:
                        wait_time = 5 * retry_count
                        print(f"采集第{page}页失败: {str(e)}")
                        print(f"等待{wait_time}秒后第{retry_count + 1}次重试...")
                        time.sleep(wait_time)
                    else:
                        print(f"第{page}页重试{max_retries}次后仍然失败，跳过该页")

        # 打印详细统计信息
        elapsed_time = time.time() - stats["start_time"]
        success_rate = (
            (stats["successful_requests"] / stats["total_requests"] * 100)
            if stats["total_requests"] > 0
            else 0
        )
        print("\n" + "=" * 50)
        print("采集任务完成!")
        print("采集统计信息:")
        print("- 请求统计:")
        print(f"  总请求次数: {stats['total_requests']}")
        print(f"  成功请求数: {stats['successful_requests']}")
        print(f"  重试次数: {stats['retries']}")
        print(f"  成功率: {success_rate:.1f}%")
        print("\n- 数据统计:")
        print(f"  处理卡片数: {stats['total_cards']}")
        print(f"  有效微博数: {stats['valid_posts']}")
        print(f"  重复微博数: {stats['duplicates']}")
        print(f"  解析错误数: {stats['errors']}")
        print("\n- 性能统计:")
        print(f"  总耗时: {elapsed_time:.1f}秒")
        print(f"  平均速度: {stats['valid_posts'] / elapsed_time:.1f}条/秒")
        print(f"  内存占用: {len(all_posts) * 1024 / (1024 * 1024):.1f}MB (估算)")
        print("=" * 50)

        return all_posts


def main():
    """测试采集功能"""
    scraper = WeiboScraper()
    posts = scraper.scrape_posts("金融科技", max_pages=2)

    # 保存采集结果
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/weibo_data.json", "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)

    print("\n数据已保存到 data/raw/weibo_data.json")
    print(f"共获取{len(posts)}条有效数据")


if __name__ == "__main__":
    main()
