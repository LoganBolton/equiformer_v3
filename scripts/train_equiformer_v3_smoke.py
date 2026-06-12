#!/usr/bin/env python
from __future__ import annotations

import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _install_torch_scatter_fallback() -> None:
    try:
        import torch_scatter  # noqa: F401

        return
    except OSError:
        pass

    import torch
    from torch_geometric.utils import scatter as pyg_scatter

    fallback = types.ModuleType("torch_scatter")
    utils = types.ModuleType("torch_scatter.utils")

    def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
        if out is not None:
            dim_size = out.size(dim)
        result = pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)
        if out is not None:
            out.zero_()
            slices = [slice(None)] * out.dim()
            slices[dim] = slice(0, result.size(dim))
            out[tuple(slices)] = result
            return out
        return result

    def segment_coo(src, index, out=None, dim_size=None, reduce="sum"):
        return scatter(src, index, dim=0, out=out, dim_size=dim_size, reduce=reduce)

    def segment_csr(src, indptr, out=None, reduce="sum"):
        chunks = []
        for start, end in zip(indptr[:-1].tolist(), indptr[1:].tolist()):
            chunk = src[start:end]
            if chunk.numel() == 0:
                chunks.append(torch.zeros_like(src[0]))
            elif reduce in {"sum", "add"}:
                chunks.append(chunk.sum(dim=0))
            elif reduce == "mean":
                chunks.append(chunk.mean(dim=0))
            elif reduce == "min":
                chunks.append(chunk.min(dim=0).values)
            elif reduce == "max":
                chunks.append(chunk.max(dim=0).values)
            else:
                raise ValueError(f"Unsupported reduce={reduce!r}")
        result = torch.stack(chunks, dim=0) if chunks else src.new_empty((0,))
        if out is not None:
            out.copy_(result)
            return out
        return result

    def broadcast(src, other, dim):
        if dim < 0:
            dim = other.dim() + dim
        if src.dim() == 1:
            for _ in range(dim):
                src = src.unsqueeze(0)
        while src.dim() < other.dim():
            src = src.unsqueeze(-1)
        return src.expand_as(other)

    fallback.scatter = scatter
    fallback.segment_coo = segment_coo
    fallback.segment_csr = segment_csr
    utils.broadcast = broadcast
    fallback.utils = utils

    sys.modules["torch_scatter"] = fallback
    sys.modules["torch_scatter.utils"] = utils


_install_torch_scatter_fallback()

# The standard FairChem auto-import hook only loads experimental modules listed
# in experimental/.include. This checkout does not have that file, so register
# the Equiformer v3 pieces needed by the smoke config explicitly.
import fairchem.core.datasets.lmdb_dataset  # noqa: F401,E402
import fairchem.core.tasks.task  # noqa: F401,E402
import experimental.models.equiformer_v3.equiformer_v3  # noqa: F401,E402
import experimental.trainers.equiformer_v3_dens_trainer  # noqa: F401,E402

from fairchem.core.common.registry import registry  # noqa: E402
from fairchem.core._cli import main  # noqa: E402

registry.register("imports_setup", True)


if __name__ == "__main__":
    main()
