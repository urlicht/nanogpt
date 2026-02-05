import numpy as np
import torch
from nanogpt.data.batch import get_batch

def _write_bin(path, data: np.ndarray) -> None:
    data.tofile(str(path))

class TestBatch:
    def test_get_batch_cpu_deterministic(self, tmp_path):
        data = np.arange(100, dtype=np.uint16)
        _write_bin(tmp_path / "train.bin", data)

        n_batch = 4
        n_block = 8
        seed = 1234

        generator = torch.Generator().manual_seed(seed)
        expected_ix = torch.randint(
            len(data) - n_block - 1,
            (n_batch,),
            generator=generator,
        ).tolist()

        torch.manual_seed(seed)
        x, y = get_batch(
            data_dir=str(tmp_path),
            split="train",
            n_batch=n_batch,
            n_block=n_block,
            device=torch.device("cpu"),
        )

        assert x.shape == (n_batch, n_block)
        assert y.shape == (n_batch, n_block)
        assert x.dtype == torch.int64
        assert y.dtype == torch.int64
        assert torch.equal(y[:, :-1], x[:, 1:])

        expected_x = torch.stack(
            [torch.from_numpy(data[i : i + n_block].astype(np.int64)) for i in expected_ix]
        )
        expected_y = torch.stack(
            [
                torch.from_numpy(data[i + 1 : i + 1 + n_block].astype(np.int64))
                for i in expected_ix
            ]
        )

        assert torch.equal(x, expected_x)
        assert torch.equal(y, expected_y)
