import unittest
from Config_and_Utils import softmax
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_softmax_input(self):
        x = np.array([8, 5, 0])
        out = softmax(x)
        assert(np.all(out > 0))

    def test_softmax(self):
        x = np.array([8, 5, 0])
        out = softmax(x)
        self.assertAlmostEqual(out.sum(), 1.0)


if __name__ == '__main__':
    unittest.main()
