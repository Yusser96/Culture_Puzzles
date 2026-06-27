import unittest
import numpy as np
import src.shared_utils.extraction as ext

class TestExtractionShapes(unittest.TestCase):
    def test_pool_mean_and_last(self):
        # hidden (batch=1, seq=3, dim=2); mask keeps tokens 0 and 2
        hidden = np.array([[[1., 1.], [9., 9.], [3., 3.]]])
        mask = np.array([[True, False, True]])
        mean = ext._pool(hidden, mask, "mean_content")
        last = ext._pool(hidden, mask, "last_content")
        np.testing.assert_allclose(mean[0], [2., 2.])
        np.testing.assert_allclose(last[0], [3., 3.])

if __name__ == "__main__":
    unittest.main()
