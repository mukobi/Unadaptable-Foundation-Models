import torch
# import datasets
from ..src.ufm.data import get_hf_data

def test_get_hf_data():
    # Test case 1: Test with "cyber" dataset identifier
    dataset_identifier = "cyber"
    batch_size = 128
    data_loader = get_hf_data(dataset_identifier, batch_size)
    assert isinstance(data_loader, torch.utils.data.DataLoader)
    assert len(data_loader.dataset) > 0

    # Test case 2: Test with "harmfulqa" dataset identifier
    dataset_identifier = "harmfulqa"
    batch_size = 128
    data_loader = get_hf_data(dataset_identifier, batch_size)
    assert isinstance(data_loader, torch.utils.data.DataLoader)
    assert len(data_loader.dataset) > 0

    # Test case 3: Test with "toxic" dataset identifier
    dataset_identifier = "toxic"
    batch_size = 128
    data_loader = get_hf_data(dataset_identifier, batch_size)
    assert isinstance(data_loader, torch.utils.data.DataLoader)
    assert len(data_loader.dataset) > 0

    # Test case 4: Test with "pile" dataset identifier
    dataset_identifier = "pile"
    batch_size = 128
    data_loader = get_hf_data(dataset_identifier, batch_size)
    assert isinstance(data_loader, torch.utils.data.DataLoader)
    assert len(data_loader.dataset) > 0

    # Test case 5: Test with unsupported dataset identifier
    dataset_identifier = "unknown"
    batch_size = 128
    try:
        data_loader = get_hf_data(dataset_identifier, batch_size)
        assert False, "Expected ValueError"
    except ValueError:
        assert True

test_get_hf_data()