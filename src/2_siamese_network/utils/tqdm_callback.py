from tqdm import tqdm
from keras.callbacks import Callback

class TQDMCallback(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.epoch_progressbar = None
        self.total_epochs = total_epochs
        
    def on_train_begin(self, logs=None):
        self.epoch_progressbar = tqdm(total=self.total_epochs, desc='Training')
 
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_progressbar.update(1)
        self.epoch_progressbar.set_postfix(logs)
        
    def on_train_end(self, batch, logs=None):
        self.epoch_progressbar.close()