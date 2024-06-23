import nni
import argparse
from deeptune_utils import DeepTune
from deeptune_utils import train_underlying
from deeptune_utils import restore_model
from deeptune_utils import IL
sys.path.append('/home/huanting/PROM/thirdpackage')
from mapie import MapieClassifier

class train_underlying():
    def __init__(self, args):
        # self.model = model
        self.args = args
        super().__init__()

    def train(self):
        if self.args=='DeepTune':
            return DeepTune_train()

    def restore_model(self):
        if self.args=='DeepTune':
            return DeepTune_load()

class ConformalP():
    def __init__(self, args):
        # self.model = model
        self.args = args
        super().__init__()

    def detect_drift(self):
        if self.args=='DeepTune':
            return DeepTune_cp()

def load_args():
    # get parameters from tuner
    params = nni.get_next_parameter()
    if params == {}:
        params = {
            "epoch": 3,
            "batch_size": 8,
            "seed": 123,
        }

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=params['seed'],
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=params['epoch'],
                        help="random seed for initialization")
    parser.add_argument("--batch_size", default=params['batch_size'], type=int,
                        help="Batch size per GPU/CPU for training.")
    args = parser.parse_args()

    # train the underlying model
    deeptune_model = DeepTune()
    deeptune_model.init(args)
    return args

def DeepTune_train():
    args=load_args()
    train_underlying(args)

def DeepTune_load():
    args=load_args()
    restore_model(DeepTune(), model_pretrained='', args=args)

def DeepTune_cp():
    args=load_args()
    restore_model(DeepTune(), model_pretrained='', args=args)
    IL(DeepTune(), args=args)