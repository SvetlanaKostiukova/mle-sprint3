from fastapi import FastAPI, Body
from fast_api_handler import FastApiHandler

# создаём FastAPI-приложение 
app = FastAPI()

# создаём обработчик запросов для API
app.handler = FastApiHandler()

@app.post("/api/churn/")
def get_prediction_for_item(user_id: str, model_params: dict):
    params = { 'user_id': user_id, 'model_params': model_params }
    return app.handler.handle(params)