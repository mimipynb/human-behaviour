"""
----------------------------------------------------------------------

initial_predictions.py 
Models trained on observed data can be used to set the expected logits for the next observing state. This script loads prediction model's for the following features:
- big 5 elements 
- mbti personality label 
TODO: additional features that has not been added
- emotions specific to user 
- dialogue responses 
NOTE: Currently, this is set as the initial expectation for every completed dialogue.

----------------------------------------------------------------------

Application:
- Runs with the huggingface_client.py such that the prediction logits for emotion scores are fetched from models on hugginface (Roberta emotions and BART classifier)
TODO: 
- Recreating the feature predicting models for the current chat sessions (SGDRegression setup)
- Probably should make an object class that utilises the initial pre-trained models relative to a certain user (me) eg: all features -> my emotion 

----------------------------------------------------------------------
Data loader / cleaner in dataUtils/prepareDataset.py 
- https://huggingface.co/datasets/Navya1602/Personality_dataset
- https://huggingface.co/datasets/kl08/myers-briggs-type-indicator 

"""
import os 
import joblib 

import numpy as np 
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer 

mbti_labels = [
    "ISTJ", "ISFJ", "INFJ", "INTJ",
    "ISTP", "ISFP", "INFP", "INTP",
    "ESTP", "ESFP", "ENFP", "ENTP",
    "ESTJ", "ESFJ", "ENFJ", "ENTJ"
]
model_card = "all-mpnet-base-v2"

class KittenBreeder:
    """Chains the Logistic regression by letter of the MBTI label."""
    
    def __init__(self, **kwargs: dict):
        for i in range(4):
            setattr(self, f"m{i}", LogisticRegression(**kwargs))

    def train(self, model: LogisticRegression, X_train: np.ndarray, y_train: np.ndarray, class_weights: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        from sklearn.metrics import classification_report
        model.fit(X_train, y_train, class_weights)
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        
    def update(self, X: np.ndarray, *ys: tuple):
        """Update the models of this class obj."""
        from sklearn.model_selection import train_test_split
        for model_num, y in enumerate(ys):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            self.train(getattr(self, f'm{model_num}'), X_train, y_train[:, 0], y_train[:, 1], X_test, y_test[:, 0])

    def predict(self, x: np.ndarray):
        """Predict mbti labels given the encoded inputs."""
        mbti_label = []
        diff_score = []
        prediction = None
        
        for model_num in range(4):
            model = getattr(self, f'm{model_num}')
            logits = model.predict_proba(x).flatten()
        
            if prediction is None:
                prediction = logits 
            else:
                prediction = np.vstack([prediction, logits])

            diff_score += [abs(np.diff(logits))[0]]
            mbti_label += [model.predict(x)[0]]

        return mbti_label, diff_score, prediction 

class Basement(SentenceTransformer):
    """Basement to encode speech."""
    def __init__(self, model: str = model_card):
        super().__init__(model)
        # Set the root directory of the project
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        big_model_path = os.path.join(root_dir, 'models', 'big5model.pkl')
        mbti_model_path = os.path.join(root_dir, 'models', 'mbtiModel.pkl')

        self.expression = joblib.load(big_model_path) 
        self.breed = joblib.load(mbti_model_path)
        self.session = []
        self.breed_belief = []
        self.temp = 1
        
    def update(self, logits: np.ndarray) -> None:
        """Updates the belief states"""
        temperature = np.min(np.max(self.breed_belief), self.temp) if len(self.breed_belief) != 0 else self.temp
        self.temp = np.maximum(temperature, 1e-3)
        exp_logits = np.exp(-logits / temperature) 
        exp_proba = exp_logits / exp_logits.sum()
        self.breed_belief.extend(exp_proba)
        assert len(np.array(self.breed_belief).shape) == 1
        
    def predict(self, query: str) -> dict:
        """Returns the predicted features on the user's input query"""
        embedding = self.encode([query])
        big_proba = self.expression.predict_proba(embedding)
        breed_label, breed_diff_score, breed_proba = self.breed.predict(embedding)
        self.update(np.array(breed_diff_score))
        self.session += [{
            'big': big_proba, 
            'mbti': tuple(zip(breed_label, breed_proba)),
            'mbti_label': "".join(breed_label)
        }]
        return self.session[-1]

if __name__ == '__main__':
    base = Basement()
    output = base.predict('I prefer to stay at home and read books honestly...')
    print(output)
    import pandas as pd 
    df = pd.DataFrame(output['mbti'], columns=['MBTI', 'data'])
    print(df)