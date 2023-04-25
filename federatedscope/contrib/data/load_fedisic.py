import os
import pickle

from federatedscope.register import register_data
from federatedscope.core.data.utils import convert_data_mode
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.data import ClientData, StandaloneDataDict
from flamby.datasets.fed_isic2019 import FedIsic2019

def load_data_from_file(config, client_cfgs=None):

    # The shape of data is expected to be:
    # (1) the data consist of all participants' data:
    # {
    #   'client_id': {
    #       'train/val/test': {
    #           'x/y': np.ndarray
    #       }
    #   }
    # }
    # (2) isolated data
    # {
    #   'train/val/test': {
    #       'x/y': np.ndarray
    #   }
    # }

    # translator = DummyDataTranslator(config, client_cfgs)
    # data = translator(data)

    # Convert `StandaloneDataDict` to `ClientData` when in distribute mode
    data_dict= {0:
              ClientData(config, train=FedIsic2019(train=True, pooled=True),
                         val=FedIsic2019(train=False, pooled=True),
                         test=FedIsic2019(train=False, pooled=True))}
    for i in range(1, 7) :
        data_dict[i]=ClientData(config, train=FedIsic2019(center=i-1, train=True),
                              val=FedIsic2019(center=i-1, train=False),
                              test=FedIsic2019(center=i-1, train=False))
    data = StandaloneDataDict(data_dict, config)
    data = convert_data_mode(data, config)

    # Restore the user-specified seed after the data generation
    setup_seed(config.seed)

    return data, config


def call_fedisic_data(config, client_cfgs):
    if config.data.type == "fedisic":
        # All the data (clients and servers) are loaded from one unified files
        data, modified_config = load_data_from_file(config, client_cfgs)
        return data, modified_config


register_data("fedisic", call_fedisic_data)
