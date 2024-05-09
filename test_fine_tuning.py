import torch
from src.ufm.fine_tuning import run_fine_tune
from src.ufm.models import HuggingFaceModel
from logging import Logger




def test_run_fine_tune():
    # Test case 0: Test with "dummy" dataset identifier and next-token-prediction task
    # model_unadapted = HuggingFaceModel()
    # model_unadapted.model = model_unadapted.model.to(torch.device('cuda'))
    # configs = {'dataset': 'dummy', 'output_dir': './results'}
    # logger = Logger('test')
    # training_task = 'next-token-prediction'
    # eval_loss = run_fine_tune(model_unadapted, configs, logger, training_task)
    # assert isinstance(eval_loss, float)

    # Test case 1: Test with "cyber" dataset identifier and next-token-prediction task
    model_unadapted = HuggingFaceModel()
    model_unadapted.model = model_unadapted.model.to(torch.device('cuda'))
    configs = {'dataset': 'cyber', 'output_dir': './results', 'num_train_epochs': 3, 'per_device_train_batch_size': 1}
    logger = Logger('test')
    training_task = 'next-token-prediction'
    eval_loss = run_fine_tune(model_unadapted, configs, logger, training_task)
    assert isinstance(eval_loss, float)

    # Test case 2: Test with "pile" dataset identifier and next-token-prediction task
    model_unadapted = HuggingFaceModel()
    model_unadapted.model = model_unadapted.model.to(torch.device('cuda'))
    configs = {'dataset': 'pile', 'output_dir': './results', 'num_train_epochs': 3, 'per_device_train_batch_size': 8}
    logger = Logger()
    training_task = 'next-token-prediction'
    eval_loss = run_fine_tune(model_unadapted, configs, logger, training_task)
    assert isinstance(eval_loss, float)

    # Test case 3: Test with unsupported dataset identifier and next-token-prediction task
    model_unadapted = HuggingFaceModel()
    model_unadapted.model = model_unadapted.model.to(torch.device('cuda'))
    configs = {'dataset': 'unknown', 'output_dir': './results', 'num_train_epochs': 3, 'per_device_train_batch_size': 8}
    logger = Logger()
    training_task = 'next-token-prediction'
    try:
        eval_loss = run_fine_tune(model_unadapted, configs, logger, training_task)
        assert False, "Expected NotImplementedError"
    except NotImplementedError:
        assert True

    # Test case 4: Test with "cyber" dataset identifier and unsupported task
    model_unadapted = HuggingFaceModel()
    model_unadapted.model = model_unadapted.model.to(torch.device('cuda'))
    configs = {'dataset': 'cyber', 'output_dir': './results', 'num_train_epochs': 3, 'per_device_train_batch_size': 8}
    logger = Logger()
    training_task = 'classification'
    try:
        eval_loss = run_fine_tune(model_unadapted, configs, logger, training_task)
        assert False, "Expected NotImplementedError"
    except NotImplementedError:
        assert True

if __name__ == "__main__":
    test_run_fine_tune()
# test_run_fine_tune()