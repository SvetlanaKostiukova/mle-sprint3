from catboost import CatBoostClassifier

class FastApiHandler:

    def __init__(self):
        # типы параметров запроса для проверки
        self.param_types = {
            "user_id": str,
            "model_params": dict
        }

        self.model_path = "models/catboost_churn_model.bin"
        self.load_churn_model(model_path=self.model_path)
        
        # необходимые параметры для предсказаний модели оттока
        self.required_model_params = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Type', 'PaperlessBilling', 'PaymentMethod', 
                'MonthlyCharges', 'TotalCharges', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'days', 'services'
            ]

    def load_churn_model(self, model_path: str):
        try:
            self.model = CatBoostClassifier()
            self.model.load_model(model_path)
        except Exception as e:
            print(f"Failed to load model: {e}")

    def churn_predict(self, model_params: dict) -> float:
        param_values_list = list(model_params.values())
        return self.model.predict_proba(param_values_list)[1]
        
    def check_required_query_params(self, query_params: dict) -> bool:
        if "user_id" not in query_params or "model_params" not in query_params:
            return False
        if not isinstance(query_params['user_id'], self.param_types['user_id']):
            return False
        if not isinstance(query_params['model_params'], self.param_types['model_params']):
            return False
        return True
    
    def check_required_model_params(self, model_params: dict) -> bool:
        if set(model_params.keys()) == set(self.required_model_params):
            return True
        return False 
    
    def validate_params(self, params: dict) -> bool:
        if self.check_required_query_params(params):
            print("All query params exist")
        else:
            print("Not all query params exist")
            return False

        if self.check_required_model_params(params["model_params"]):
            print("All model params exist")
        else:
            print("Not all model params exist")
            return False
        return True
        
    def handle(self, params):
        try:
            if not self.validate_params(params):
                response = {'Error': 'Problem with parameters'}
            else:
                model_params = params['model_params']
                user_id = params['user_id']
                probability = self.churn_predict(model_params)
                response = {'user_id': user_id, 'probability': probability, 'is_churn': int(probability > 0.5)}
        except Exception as e:
            print(f"Error while handling request: {e}")
        else:
            return response
        
if __name__ == "__main__":

    # создаём тестовый запрос
    test_params = {
        'user_id': '123',
        'model_params': {
              'gender': 1.0,
              'SeniorCitizen': 0.0,
              'Partner': 0.0,
              'Dependents': 0.0,
              'Type': 0.5501916796819537,
              'PaperlessBilling': 1.0,
              'PaymentMethod': 0.2192247621752094,
              'MonthlyCharges': 50.8,
              'TotalCharges': 288.05,
              'MultipleLines': 0.0,
              'InternetService': 0.3437455629703251,
              'OnlineSecurity': 0.0,
              'OnlineBackup': 0.0,
              'DeviceProtection': 0.0,
              'TechSupport': 1.0,
              'StreamingTV': 0.0,
              'StreamingMovies': 0.0,
              'days': 245.0,
              'services': 2.0
        }
    }

    # создаём обработчик запросов для API
    handler = FastApiHandler()

    # делаем тестовый запрос
    response = handler.handle(test_params)
    print(f"Response: {response}")