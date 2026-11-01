import gc
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch

from .similarity import unpack_128bit_to_binary


NEGATIVE_SAMPLING_POLICIES = ("uniform", "exclude_known_positive")


class KnownPositiveIndex:
    """Complete profile-to-AS CSR membership, including every hop and weight."""

    def __init__(self, profile_ids, as_ids, num_profiles, num_asns, device="cpu"):
        self.num_profiles, self.num_asns = int(num_profiles), int(num_asns)
        self.device = torch.device(device)
        pids = torch.as_tensor(profile_ids).to(device="cpu", dtype=torch.long)
        aids = torch.as_tensor(as_ids).to(device="cpu", dtype=torch.long)
        keys = pids * self.num_asns + aids
        if len(keys) > 1 and not bool((keys[1:] > keys[:-1]).all()):
            keys = torch.unique(keys, sorted=True)
        self.known_pairs = keys.numel()
        sorted_pids = torch.div(keys, self.num_asns, rounding_mode="floor")
        self.counts_cpu = torch.bincount(sorted_pids, minlength=self.num_profiles)
        self.starts_cpu = self.counts_cpu.cumsum(0) - self.counts_cpu
        self.counts = self.counts_cpu.to(self.device)
        self.starts = self.starts_cpu.to(self.device)
        self.as_ids = keys.remainder(self.num_asns).to(device=self.device, dtype=torch.int32)
        self._search_steps = int(self.counts_cpu.max()).bit_length()

    @classmethod
    def from_triplets(cls, triplets, num_profiles, num_asns, device="cpu"):
        return cls(triplets[:, 0], triplets[:, 1], num_profiles, num_asns, device)

    @torch.no_grad()
    def contains(self, profile_ids, as_ids):
        pids, aids = torch.broadcast_tensors(
            torch.as_tensor(profile_ids, dtype=torch.long, device=self.device),
            torch.as_tensor(as_ids, dtype=torch.long, device=self.device),
        )
        shape = pids.shape
        pids, aids = pids.reshape(-1), aids.reshape(-1)
        low = self.starts[pids].clone()
        ends = low + self.counts[pids]
        high = ends.clone()
        for _ in range(self._search_steps):
            active = low < high
            middle = (low + high) // 2
            values = self.as_ids[middle.clamp(max=self.known_pairs - 1)]
            smaller = active & (values < aids)
            low = torch.where(smaller, middle + 1, low)
            high = torch.where(active & ~smaller, middle, high)
        found = (low < ends) & (self.as_ids[low.clamp(max=self.known_pairs - 1)] == aids)
        return found.reshape(shape)

    @torch.no_grad()
    def sample(self, profile_ids, num_samples, generator=None):
        pids = torch.as_tensor(profile_ids, dtype=torch.long, device=self.device)
        candidates = torch.randint(self.num_asns, (len(pids), num_samples),
                                   device=self.device, generator=generator)
        expanded_pids = pids[:, None].expand_as(candidates)
        rejected = self.contains(expanded_pids, candidates)
        flat_candidates = candidates.reshape(-1)
        positions = torch.nonzero(rejected.reshape(-1), as_tuple=True)[0]
        while positions.numel():
            redrawn = torch.randint(self.num_asns, (positions.numel(),),
                                    device=self.device, generator=generator)
            flat_candidates[positions] = redrawn
            positions = positions[self.contains(pids[positions // num_samples], redrawn)]
        return candidates


def sample_negative_as(profile_ids, num_samples, num_asns, *, policy="uniform",
                       known_positive_index=None, generator=None):
    """Draw with replacement, rejecting complete known positives when requested."""
    if policy == "exclude_known_positive":
        return known_positive_index.sample(profile_ids, num_samples, generator)
    pids = torch.as_tensor(profile_ids)
    return torch.randint(num_asns, (len(pids), int(num_samples)), device=pids.device, generator=generator)


class CompactSampler:
    def __init__(self, triplets_tensor: torch.Tensor, num_profiles: int, device: torch.device):
        t0 = time.time()
        sorted_indices = torch.argsort(triplets_tensor[:, 0])
        sorted_triplets = triplets_tensor[sorted_indices]

        profile_ids = sorted_triplets[:, 0].to(dtype=torch.long, device="cpu")
        as_ids = sorted_triplets[:, 1].to(dtype=torch.long, device="cpu")
        hop_values = sorted_triplets[:, 2].to(dtype=torch.float32, device="cpu")

        self.val_as = as_ids.to(device=device, dtype=torch.long)
        self.val_hops = hop_values.clamp(min=0, max=255).to(device=device, dtype=torch.uint8)
        if sorted_triplets.shape[1] >= 4:
            self.val_weights = sorted_triplets[:, 3].to(device=device, dtype=torch.float32)
        else:
            self.val_weights = torch.ones_like(self.val_hops, dtype=torch.float32, device=device)

        full_counts = torch.bincount(profile_ids, minlength=num_profiles)

        self.counts = full_counts.to(device=device, dtype=torch.long)
        self.starts = (torch.cumsum(full_counts, dim=0) - full_counts).to(device=device, dtype=torch.long)
        self.device = device

        del sorted_indices, sorted_triplets, profile_ids, as_ids, hop_values, full_counts
        gc.collect()
        print(f"Sampler initialized in {time.time() - t0:.2f}s")

    def sample(self, batch_profile_indices: torch.Tensor, num_samples: int = 4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = batch_profile_indices.size(0)
        starts = self.starts[batch_profile_indices]
        counts = self.counts[batch_profile_indices]
        safe_counts = torch.clamp(counts, min=1).unsqueeze(1)

        rand_offsets = torch.randint(0, 1_000_000, (bsz, num_samples), device=self.device) % safe_counts
        global_indices = starts.unsqueeze(1) + rand_offsets
        flat_indices = global_indices.view(-1)

        sampled_as = self.val_as[flat_indices].view(bsz, num_samples)
        sampled_hops = self.val_hops[flat_indices].view(bsz, num_samples).float()
        sampled_weights = self.val_weights[flat_indices].view(bsz, num_samples)
        return sampled_as, sampled_hops, sampled_weights


def load_training_artifacts(data_dir: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    with open(data_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    raw_fp = torch.load(data_dir / "profile_fingerprints.pt", map_location=device)
    fingerprints = unpack_128bit_to_binary(raw_fp.to(device))

    triplets = torch.load(data_dir / "profile_triplets.pt", map_location="cpu")

    return {
        "meta": meta,
        "triplets": triplets,
        "fingerprints": fingerprints,
    }
