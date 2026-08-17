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


def _probe(description: str, call) -> bool:
    """Return True if a compiled extension is present AND runs on this GPU.

    A wheel built without the local GPU's architecture imports perfectly well
    and only fails when a kernel is launched ("no kernel image is available for
    execution on the device"), so importing is not evidence enough. Probe on the
    device that training will actually use.
    """
    import torch

    try:
        call(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    except Exception as error:  # noqa: BLE001 - any failure means "unusable here"
        print(
            f"[compat] {description} is unavailable, substituting a pure-torch "
            f"equivalent. Results are unaffected but this is not the reference "
            f"code path. Reason: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return False
    return True


def _install_torch_scatter_fallback() -> None:
    def probe(device):
        import torch
        import torch_scatter

        source = torch.ones(4, 2, device=device)
        index = torch.zeros(4, dtype=torch.long, device=device)
        torch_scatter.scatter(source, index, dim=0)

    if _probe("torch-scatter", probe):
        return

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


def _install_radius_graph_fallback() -> None:
    """Provide a pure-torch ``radius_graph`` when no compiled backend exists.

    ``torch_geometric.nn.radius_graph`` needs either torch-cluster or pyg-lib,
    whose CUDA kernels have to be built for the exact GPU architecture in use.
    Rather than requiring every machine to compile them, fall back to a dense
    implementation. Molecules here are tiny (5 atoms), so the cost is
    negligible and the result matches torch-cluster edge for edge.
    """
    import torch
    import torch_geometric
    import torch_geometric.nn

    def probe(device):
        positions = torch.zeros(4, 3, device=device)
        batch = torch.zeros(4, dtype=torch.long, device=device)
        torch_geometric.nn.radius_graph(positions, r=1.0, batch=batch, max_num_neighbors=4)

    if _probe("torch_geometric.nn.radius_graph", probe):
        return

    def radius_graph(
        x,
        r,
        batch=None,
        loop=False,
        max_num_neighbors=32,
        flow="source_to_target",
        num_workers=1,
        batch_size=None,
    ):
        assert flow in ("source_to_target", "target_to_source")
        num_nodes = x.size(0)
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)

        with torch.no_grad():
            num_graphs = int(batch.max().item()) + 1 if num_nodes else 0
            counts = torch.bincount(batch, minlength=num_graphs)
            width = int(counts.max().item()) if num_graphs else 0
            offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
            node_index = torch.arange(num_nodes, device=x.device)
            slot = node_index - offsets[batch]

            # Scatter the flat node list into a [graph, slot, 3] dense block so
            # every graph's pairwise distances are computed independently.
            positions = x.new_zeros((num_graphs, width, x.size(1)))
            positions[batch, slot] = x.detach()
            flat_index = torch.full(
                (num_graphs, width), -1, dtype=torch.long, device=x.device
            )
            flat_index[batch, slot] = node_index
            occupied = flat_index >= 0

            distance = torch.cdist(
                positions, positions, compute_mode="donot_use_mm_for_euclid_dist"
            )
            mask = (distance <= r) & occupied.unsqueeze(1) & occupied.unsqueeze(2)
            if not loop:
                identity = torch.eye(width, dtype=torch.bool, device=x.device)
                mask &= ~identity.unsqueeze(0)
            if max_num_neighbors is not None:
                # torch-cluster keeps the first neighbours it encounters in
                # index order rather than the nearest ones; match that.
                mask &= mask.cumsum(dim=2) <= max_num_neighbors

            graph, center, neighbor = mask.nonzero(as_tuple=True)
            target = flat_index[graph, center]
            source = flat_index[graph, neighbor]

        if flow == "source_to_target":
            return torch.stack([source, target], dim=0)
        return torch.stack([target, source], dim=0)

    torch_geometric.nn.radius_graph = radius_graph
    torch_geometric.nn.pool.radius_graph = radius_graph


_install_torch_scatter_fallback()
_install_radius_graph_fallback()

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
