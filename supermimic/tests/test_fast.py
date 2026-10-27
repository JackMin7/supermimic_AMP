import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from rsl_rl.utils.utils import Normalizer
from supermimic.core.normalization import TensorAMPNormalizer
from supermimic.core.checkpoint import atomic_save


class FastTest(unittest.TestCase):
    def test_amp_statistics_match_native_updates_and_reload(self):
        torch.manual_seed(4);native=Normalizer(5);tensor=TensorAMPNormalizer(native,'cpu')
        for _ in range(10):
            raw=torch.randn(64,5)*2+1
            expected=native.normalize_torch(raw,'cpu');actual=tensor.normalize_torch(raw)
            torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-6)
            # Preserve native training's normalized-current-feature update convention.
            native.update(expected.numpy());tensor.update_tensor(actual)
        exported=tensor.native()
        np.testing.assert_allclose(exported.mean,native.mean,rtol=2e-6,atol=2e-6)
        np.testing.assert_allclose(exported.var,native.var,rtol=2e-6,atol=2e-6)
        self.assertEqual(exported.count,native.count)
        reloaded=TensorAMPNormalizer(exported,'cpu')
        torch.testing.assert_close(reloaded.normalize_torch(raw),tensor.normalize_torch(raw))

    def test_failed_atomic_checkpoint_keeps_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'last.pt';atomic_save({'update':100},path)
            with patch('torch.save',side_effect=RuntimeError('simulated disk failure')):
                with self.assertRaises(RuntimeError):atomic_save({'update':200},path)
            self.assertEqual(torch.load(path,weights_only=False)['update'],100)
            self.assertFalse(path.with_name('last.pt.tmp').exists())


if __name__=='__main__':unittest.main()
