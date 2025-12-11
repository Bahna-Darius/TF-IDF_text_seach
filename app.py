from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from src.schemas import ArticleInput, UserQuery, ClassificationResponse, RecommendationResponse
from src import inference
import uvicorn



@asynccontextmanager
async def lifespan(app: FastAPI):
    await inference.load_resources()
    yield
    print("🛑 Shutting down...")


app = FastAPI(
    title="News AI Microservice",
    description="Modular API for News Classification & Retrieval",
    version="2.0",
    lifespan=lifespan
)



@app.get("/")
def home():
    return {"status": "System Operational", "mode": "Production"}


@app.post("/classify", response_model=ClassificationResponse)
async def classify_endpoint(payload: ArticleInput):
    try:
        category, confidence = await inference.predict_category_logic(payload.title, payload.content)

        return {
            "category": category,
            "confidence_score": confidence
        }

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_endpoint(payload: UserQuery):
    try:
        recs = await inference.get_recommendations_logic(payload.text, payload.top_n)

        return {
            "results": recs
        }

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
