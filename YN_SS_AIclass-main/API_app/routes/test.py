from fastapi import APIRouter
import torch

from API_app.model.dataclass import DataInput, PredictOutput
from core.config import Config

from ML_BASE.ML_operate.operator import prediction, validation, training
from ML_BASE.ML_models.LSTM_ex01 import RNN
from ML_BASE.ML_main import ML_runway

modelConfig = Config("NN01",0)

model = RNN(
        modelConfig.input_size, 
        modelConfig.hidden_sizes, 
        modelConfig.seq_len, 
        dropout=modelConfig.dro, 
        output_size=1)

model.eval()

test = APIRouter(prefix='/test')

@test.get('/', tags=['test'])
async def start_test():
    return {'msg' : 'this is test'}

@test.post('/predict', tags=['test'], response_model=PredictOutput)
async def start_test(request_Input:DataInput):
    NM = request_Input.NM
    model = ML_runway(NM)
    result =  model.predict()
    return {'prediction' : result}    