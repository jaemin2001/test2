from ML_BASE.ML_operate.operator import prediction, validation, training
from ML_BASE.ML_models.LSTM_ex01 import RNN
from ML_BASE.data_provider.ML_DataReady import dataloader

class ML_runway:
    def __init__(self, NM, device="cpu") -> None:
       self.model = MODEL_CLASS[NM]["model"](
          MODEL_CONFIG[NM].get("input_size"),
          MODEL_CONFIG[NM].get("hidden_sizes"),
          MODEL_CONFIG[NM].get("seq_len"),
          MODEL_CONFIG[NM].get("dropout"),
          MODEL_CONFIG[NM].get("output_size")
          )
       self.loader = MODEL_CLASS[NM]["loader"] 
       self.device = device
       self.predicter = MODEL_CLASS[NM]["predicter"]
    
    def predict(self):
        return self.predicter(self.model, self.loader, self.device)
        

MODEL_CLASS = { 
            "NN01" : {
                        "model" : RNN,
                        "loader" : dataloader,
                        "trainer" : training,
                        "validater" : validation,
                        "predicter" : prediction,
                     }
            }

MODEL_CONFIG = {
                        "NN01" : {
                                    "batch_size" : 16,
                                    "hidden_sizes" : [288, 192, 144, 96, 32],
                                    "max_learning_rate" : 0.001,
                                    "epochs" : 100,
                                    "input_size" : 0,
                                    "seq_len" : 0,
                                    "dropout" : 0.5,
                                    "output_size" : 1,
                                    "save_path" : "./",
                                }
                   }     

