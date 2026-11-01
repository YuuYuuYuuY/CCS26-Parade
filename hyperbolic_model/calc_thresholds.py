import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).parent))
from model import HyperbolicRoutingModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.core.data import KnownPositiveIndex, NEGATIVE_SAMPLING_POLICIES, sample_negative_as

PERC_POS_HIGH_SIGMA = 99.95
PERC_NEG_LOW_SIGMA = 1.6


def find_min_error_threshold(pos_dists, neg_dists, n_grid=10000):
    """Minimize empirical FPR + FNR for distances at or below the positive threshold."""
    pos = np.asarray(pos_dists)
    neg = np.asarray(neg_dists)
    pos_sorted = np.sort(pos)
    neg_sorted = np.sort(neg)

    combined_size = pos.size + neg.size
    thresholds = None

    if combined_size <= 100000:
        combined = np.unique(np.concatenate([pos_sorted, neg_sorted]))
        if combined.size <= n_grid:
            thresholds = combined

    if thresholds is None:
        lo = float(min(pos_sorted[0], neg_sorted[0]))
        hi = float(max(pos_sorted[-1], neg_sorted[-1]))
        thresholds = np.linspace(lo, hi, n_grid)

    pos_counts = np.searchsorted(pos_sorted, thresholds, side='right')
    neg_counts = np.searchsorted(neg_sorted, thresholds, side='right')

    cdf_pos = pos_counts / pos_sorted.size
    cdf_neg = neg_counts / neg_sorted.size

    fnr = 1.0 - cdf_pos
    fpr = cdf_neg

    errs = fnr + fpr
    idx = int(np.argmin(errs))
    best_t = float(thresholds[idx])
    stats = {'fnr': float(fnr[idx]), 'fpr': float(fpr[idx]), 'error_sum': float(errs[idx])}

    return best_t, stats

class ThresholdCalculator:
    def __init__(
        self,
        data_dir,
        model_dir,
        embed_dim,
        num_vectors,
        device=None,
        seed=42,
        update_data_meta=True,
        negative_sampling="uniform",
        positive_percentile=PERC_POS_HIGH_SIGMA,
    ):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.embed_dim = embed_dim
        self.num_vectors = num_vectors
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = int(seed)
        self.positive_percentile = positive_percentile
        self.update_data_meta = update_data_meta
        self.negative_sampling = negative_sampling
        self.known_positive_index = None
        self.rib_positive_index = None

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.calculated_thresholds = {}

        self._load_resources()

    def _load_resources(self):
        with open(self.data_dir / "meta.json", "r") as f:
            self.meta = json.load(f)

        self.model = HyperbolicRoutingModel(
            num_as=self.meta["num_asns"],
            num_profiles=self.meta["num_profiles"],
            embed_dim=self.embed_dim,
            num_vectors=self.num_vectors
        ).to(self.device)

        model_path = self.model_dir / "best_model.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.eval()

        self.as_emb = self.model.as_embedding.detach()
        self.prof_emb = self.model.profile_embedding.detach()

        self.triplets = torch.load(self.data_dir / "profile_triplets.pt", map_location="cpu", mmap=True, weights_only=True)
        if self.negative_sampling == "exclude_known_positive":
            self.known_positive_index = KnownPositiveIndex.from_triplets(
                self.triplets, self.meta["num_profiles"], self.meta["num_asns"], device=self.device)
        total_samples = len(self.triplets)
        weights = self.triplets[:, 3].to(dtype=torch.float32)
        rib_mask = torch.isclose(weights, torch.ones_like(weights), atol=1e-6)
        kept_samples = int(rib_mask.sum())
        if kept_samples != total_samples:
            self.triplets = self.triplets[rib_mask]
        print(f"Triplets filtered by weight==1 (RIB): kept {kept_samples}/{total_samples}")
        if self.negative_sampling == "exclude_known_positive":
            self.rib_positive_index = self.known_positive_index if kept_samples == total_samples else (
                KnownPositiveIndex.from_triplets(
                    self.triplets, self.meta["num_profiles"], self.meta["num_asns"], device="cpu"))
        else:
            self.triplets_df = pd.DataFrame(self.triplets[:, :3].numpy(), columns=['pid', 'aid', 'hop'])

    def _calc_dist_indices(self, idx_a, idx_b, emb_a, emb_b, batch_size=8192):
        """Batch indexed embedding distances and minimize over profile vectors."""
        res = []
        idx_a = torch.as_tensor(idx_a, device=self.device, dtype=torch.long)
        idx_b = torch.as_tensor(idx_b, device=self.device, dtype=torch.long)

        with torch.no_grad():
            for i in range(0, len(idx_a), batch_size):
                b_idx_a = idx_a[i:i+batch_size]
                b_idx_b = idx_b[i:i+batch_size]

                b_a = emb_a[b_idx_a]
                b_b = emb_b[b_idx_b]

                if b_a.dim() == 3 and b_b.dim() == 2:
                    b_b = b_b.unsqueeze(1)
                elif b_a.dim() == 2 and b_b.dim() == 3:
                    b_a = b_a.unsqueeze(1)

                d = self.model.manifold.dist(b_a, b_b)

                if d.dim() > 1:
                    d, _ = d.min(dim=1)

                res.append(d.cpu())
        return torch.cat(res).numpy()

    def run(self):
        mask_pos = (self.triplets[:, 2] < 8)
        pos_triplets = self.triplets[mask_pos]
        pos_pids = pos_triplets[:, 0].to(dtype=torch.long)
        pos_aids = pos_triplets[:, 1].to(dtype=torch.long)
        pos_dists = self._calc_dist_indices(
            pos_pids,
            pos_aids,
            self.prof_emb,
            self.as_emb
        )

        n_neg = min(len(pos_dists), 10000000)
        neg_pids = torch.randint(0, len(self.prof_emb), (n_neg,), device=self.device)
        neg_aids = sample_negative_as(
            neg_pids, 1, len(self.as_emb), policy=self.negative_sampling,
            known_positive_index=self.known_positive_index).reshape(-1)

        neg_dists = self._calc_dist_indices(
            neg_pids,
            neg_aids,
            self.prof_emb,
            self.as_emb
        )

        t1_err, _ = find_min_error_threshold(pos_dists, neg_dists)
        t1_pos_sigma = float(np.percentile(pos_dists, self.positive_percentile))
        t1_neg_sigma = float(np.percentile(neg_dists, PERC_NEG_LOW_SIGMA))

        t1_final = min(t1_neg_sigma, max(t1_pos_sigma, t1_err))

        self.calculated_thresholds['threshold_profile_as'] = t1_final
        self.calculated_thresholds['threshold_profile_as_details'] = {
            'pos_sigma': t1_pos_sigma,
            'neg_sigma': t1_neg_sigma,
            'min_err': t1_err
        }

        mask_origin = (self.triplets[:, 2] == 0)
        origin_triplets = self.triplets[mask_origin]
        origin_pids = origin_triplets[:, 0].to(dtype=torch.long)
        origin_aids = origin_triplets[:, 1].to(dtype=torch.long)
        origin_dists = self._calc_dist_indices(
            origin_pids,
            origin_aids,
            self.prof_emb,
            self.as_emb
        )
        t2_err, _ = find_min_error_threshold(origin_dists, neg_dists)
        t2_pos_sigma = float(np.percentile(origin_dists, self.positive_percentile))
        t2_neg_sigma = float(np.percentile(neg_dists, PERC_NEG_LOW_SIGMA))
        t2_final = min(t2_neg_sigma, max(t2_pos_sigma, t2_err))

        self.calculated_thresholds['threshold_profile_origin'] = t2_final
        self.calculated_thresholds['threshold_profile_origin_details'] = {
            'pos_sigma': t2_pos_sigma,
            'neg_sigma': t2_neg_sigma,
            'min_err': t2_err
        }

        if self.negative_sampling == "exclude_known_positive":
            origin_df = pd.DataFrame(origin_triplets[:, :2].cpu().numpy(), columns=['pid', 'aid'])
            pid_to_origins = origin_df.groupby('pid')['aid'].apply(list).to_dict()
            valid_pids = list(pid_to_origins)
            rib_as_ids = self.rib_positive_index.as_ids.cpu()
        else:
            origin_df = self.triplets_df[self.triplets_df['hop'] == 0]
            pid_to_origins = origin_df.groupby('pid')['aid'].apply(list).to_dict()
            pid_to_associated = self.triplets_df.groupby('pid')['aid'].apply(set).to_dict()
            valid_pids = [p for p in pid_to_origins.keys() if p in pid_to_associated]
        MAX_SAMPLES = 100000
        if len(valid_pids) > MAX_SAMPLES:
            sampled_pids = np.random.choice(valid_pids, MAX_SAMPLES, replace=False)
        else:
            sampled_pids = valid_pids

        assoc_dists = []
        unrelated_dists = []

        as_emb_device = self.as_emb.to(self.device)
        num_as = len(self.as_emb)

        for pid in sampled_pids:
            origins = pid_to_origins[pid]
            if self.negative_sampling == "exclude_known_positive":
                pid_int = int(pid)
                start = int(self.rib_positive_index.starts_cpu[pid_int])
                count = int(self.rib_positive_index.counts_cpu[pid_int])
                associated = rib_as_ids[start:start + count].long().tolist()
            else:
                associated = list(pid_to_associated[pid])
            origin_vecs = as_emb_device[origins]
            assoc_vecs = as_emb_device[associated]
            with torch.no_grad():
                dists = self.model.manifold.dist(assoc_vecs.unsqueeze(1), origin_vecs.unsqueeze(0))
                min_dists, _ = dists.min(dim=1)
                assoc_dists.extend(min_dists.cpu().numpy())

            needed = 25
            if self.negative_sampling == "exclude_known_positive":
                unrelated_indices = sample_negative_as(
                    torch.tensor([int(pid)], device=self.device), needed, num_as,
                    policy=self.negative_sampling, known_positive_index=self.known_positive_index).flatten().tolist()
            else:
                candidates_pool = np.random.randint(0, num_as, needed * 5)
                unrelated_indices = [c for c in candidates_pool if c not in pid_to_associated[pid]]
                if len(unrelated_indices) > needed:
                    unrelated_indices = unrelated_indices[:needed]

            if unrelated_indices:
                unrelated_vecs = as_emb_device[unrelated_indices]
                with torch.no_grad():
                     dists_unrel = self.model.manifold.dist(unrelated_vecs.unsqueeze(1), origin_vecs.unsqueeze(0))
                     min_unrel, _ = dists_unrel.min(dim=1)
                     unrelated_dists.extend(min_unrel.cpu().numpy())

        t3_err, _ = find_min_error_threshold(assoc_dists, unrelated_dists)
        t3_neg_sigma = float(np.percentile(unrelated_dists, PERC_NEG_LOW_SIGMA))
        t3_pos_sigma = float(np.percentile(assoc_dists, self.positive_percentile))
        t3_final = min(t3_neg_sigma, max(t3_pos_sigma, t3_err))

        self.calculated_thresholds['threshold_as_origin'] = t3_final
        self.calculated_thresholds['threshold_as_origin_details'] = {
            'pos_sigma': t3_pos_sigma,
            'neg_sigma': t3_neg_sigma,
            'min_err': t3_err
        }

        self._persist_thresholds()

    def _persist_thresholds(self):
        if self.update_data_meta:
            meta_path = self.data_dir / "meta.json"

            with open(meta_path, "r") as f:
                meta = json.load(f)

            meta.update(self.calculated_thresholds)

            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=4)

        self.model_dir.mkdir(parents=True, exist_ok=True)
        with open(self.model_dir / "thresholds.json", "w") as f:
            json.dump(self.calculated_thresholds, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--model_dir", type=Path, default=Path("saved_models"))
    parser.add_argument("--embed_dim", type=int, default=24)
    parser.add_argument("--num_vectors", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-percentile", type=float, default=PERC_POS_HIGH_SIGMA,
                        help="Positive distance percentile in [0, 100] (default: 99.95)")
    parser.add_argument("--negative-sampling", choices=NEGATIVE_SAMPLING_POLICIES, default="uniform")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Write thresholds.json only under model_dir; leave data/meta.json unchanged.",
    )
    args = parser.parse_args()
    calc = ThresholdCalculator(
        args.data_dir,
        args.model_dir,
        args.embed_dim,
        args.num_vectors,
        seed=args.seed,
        update_data_meta=not args.model_only,
        negative_sampling=args.negative_sampling,
        positive_percentile=args.positive_percentile,
        device=None if args.device == "auto" else torch.device(args.device),
    )
    calc.run()
