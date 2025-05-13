import torch
import unittest
from models.omega_layer import OmegaLayer
import math

class TestOmegaLayer(unittest.TestCase):

    def setUp(self):
        self.in_dim = 5
        self.out_dim = 100
        self.batch_size = 10
        self.sigma = 1.0

        self.layer = OmegaLayer(self.in_dim, self.out_dim, sigma=self.sigma)
        self.input_tensor = torch.randn(self.batch_size, self.in_dim)

    def test_output_shape(self):
        output = self.layer(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.out_dim), "Output shape mismatch.")

    def test_value_range(self):
        output = self.layer(self.input_tensor)
        max_val = math.sqrt(2.0 / self.out_dim)
        self.assertTrue(torch.all(output <= max_val))
        self.assertTrue(torch.all(output >= -max_val))

    def test_forward_consistency(self):
        output1 = self.layer(self.input_tensor)
        output2 = self.layer(self.input_tensor)
        self.assertTrue(torch.allclose(output1, output2), "Layer should be deterministic (no dropout etc.)")

if __name__ == "__main__":
    unittest.main()