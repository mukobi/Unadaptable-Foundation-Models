from typing import Tuple

import datasets as huggingface_datasets
import torch.utils.data
import torch
import unittest
from data import get_huggingface_data

class TestData(unittest.TestCase):
    def test_get_huggingface_data(self):
        dataset_path = 'cais/wmdp-corpora'
        dataset_subset = 'cyber-forget-corpus'
        batch_size = 128
        test_batch_size = 100

        train_loader, test_loader = get_huggingface_data(dataset_path, dataset_subset, batch_size, test_batch_size)

        # Assert that the train_loader and test_loader are instances of torch.utils.
        # data.DataLoader
        self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(test_loader, torch.utils.data.DataLoader) 

        # Assert that the train_loader and test_loader have the expected batch sizes
        self.assertEqual(len(train_loader.batch_sampler), batch_size)
        self.assertEqual(len(test_loader.batch_sampler), test_batch_size)

        # Add more assertions as needed to validate the behavior of the function

if __name__ == '__main__':
    unittest.main()

