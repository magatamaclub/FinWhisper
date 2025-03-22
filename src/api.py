import os
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from preprocessor import TextPreprocessor
from trainer import TextClassifier

# 初始化API应用
app = FastAPI(
    title="中文文本分类API",
    description="基于机器学习的中文文本分类服务",
    version="1.0.0",
)

# 初始化预处理器和分类器
preprocessor = TextPreprocessor()
classifier = TextClassifier()

# 检查模型文件是否存在
if os.path.exists(os.path.join("models", "classifier.joblib")):
    classifier.load_model()
else:
    raise RuntimeError("模型文件不存在，请先运行trainer.py训练模型")


class TextRequest(BaseModel):
    text: str


class BatchTextRequest(BaseModel):
    texts: List[str]


class PredictionResponse(BaseModel):
    text: str
    category: str
    confidence: float
    features: Dict


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: TextRequest):
    """预测单条文本的类别"""
    try:
        # 文本预处理
        cleaned_text = preprocessor.clean_text(request.text)
        words = preprocessor.segment_text(cleaned_text)
        features = preprocessor.extract_features(cleaned_text)

        # 模型预测
        prediction = classifier.predict([cleaned_text], [features])[0]

        # 获取预测概率
        confidence = float(
            classifier.classifier.decision_function(
                classifier.vectorizer.transform([cleaned_text]).toarray()
            ).max()
        )

        return {
            "text": request.text,
            "category": prediction,
            "confidence": confidence,
            "features": features,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测过程出错: {str(e)}")


@app.post("/batch-predict")
async def predict_batch(request: BatchTextRequest):
    """批量预测多条文本的类别"""
    try:
        results = []
        for text in request.texts:
            # 文本预处理
            cleaned_text = preprocessor.clean_text(text)
            words = preprocessor.segment_text(cleaned_text)
            features = preprocessor.extract_features(cleaned_text)

            # 模型预测
            prediction = classifier.predict([cleaned_text], [features])[0]

            # 获取预测概率
            confidence = float(
                classifier.classifier.decision_function(
                    classifier.vectorizer.transform([cleaned_text]).toarray()
                ).max()
            )

            results.append(
                {
                    "text": text,
                    "category": prediction,
                    "confidence": confidence,
                    "features": features,
                }
            )

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量预测过程出错: {str(e)}")


@app.get("/categories")
async def get_categories():
    """获取所有可能的分类类别"""
    try:
        categories = classifier.label_encoder.classes_.tolist()
        return {"categories": categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取分类类别出错: {str(e)}")


@app.get("/health")
async def health_check():
    """API服务健康检查"""
    return {
        "status": "healthy",
        "model_loaded": os.path.exists(os.path.join("models", "classifier.joblib")),
    }


def main():
    """启动API服务"""
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
