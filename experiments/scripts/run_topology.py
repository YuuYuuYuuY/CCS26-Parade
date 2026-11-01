#!/usr/bin/env python3
"""Generate PM/RL samples, score them, and draw the Figure 9 heatmaps."""
import argparse
import bz2
from collections import Counter, defaultdict
import ctypes
from dataclasses import dataclass
import gzip
import json
import mmap
from pathlib import Path
import random
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from hyperbolic_model.detect import load_resources, load_thresholds, score_candidates

SOURCE = Path("/data/parade_ae/dataset/topology")


TIERS = ("[1,500]", "(500,2000]", "(2000,5000]", "(5000,15000]", ">15000")
TIMESTAMP = 1767225600  # January 1, 2026, 00:00 UTC.
MANIFEST_COLUMNS = [
    "sample_id", "scenario", "prefix", "au", "ad", "depth", "rank", "rank_tier",
    "source_aspath", "synthetic_aspath", "timestamp", "asn", "origin", "level",
    "topology_level", "is_origin_asn", "is_blackhole_route", "is_ip_leasing",
    "upstream_asns", "affected_asns", "note",
]
_INDEX_HEADER = struct.Struct("<5QI")
_INDEX_RANGE = struct.Struct("<iQQ")


@dataclass
class PrefixPaths:
    paths: object


class _FullPaths:
    def __init__(self, mapped, indexed, offsets_start, count):
        self.mapped, self.indexed, self.offsets_start, self.count = mapped, indexed, offsets_start, count

    def __len__(self):
        return self.count

    def __iter__(self):
        for number in range(self.count):
            offset, = struct.unpack_from("<Q", self.indexed, self.offsets_start + number*8)
            size = self.mapped[offset+4]
            offset += 5
            path = struct.unpack_from(f"<{size}I", self.mapped, offset)
            yield tuple(asn for number, asn in enumerate(path) if not number or asn != path[number-1])


def load_reference(index_path, *, reference_paths, prefix_map):
    index_path, source = Path(index_path), Path(reference_paths)
    if not index_path.exists():
        index_path.parent.mkdir(parents=True, exist_ok=True)
        binary = index_path.parent / "reference_index_builder"
        if not binary.exists():
            subprocess.run(["g++", "-O3", "-std=c++17", str(Path(__file__).with_name("reference_index.cpp")),
                            "-o", str(binary)], check=True)
        subprocess.run([str(binary.resolve()), str(source), str(index_path)], check=True)
    prefix_by_id = {int(index): prefix for prefix, index in prefix_map.items()}
    with source.open("rb") as raw:
        mapped = mmap.mmap(raw.fileno(), 0, access=mmap.ACCESS_READ)
    with index_path.open("rb") as stream:
        stream.seek(8)
        size, records, prefixes, num_asns, num_edges, name_size = _INDEX_HEADER.unpack(stream.read(_INDEX_HEADER.size))
        stream.seek(name_size, 1)
        indexed = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
        offsets_start = stream.tell() + prefixes*_INDEX_RANGE.size
        entries = {}
        for _ in range(prefixes):
            prefix_id, start, count = _INDEX_RANGE.unpack(stream.read(_INDEX_RANGE.size))
            entries[prefix_by_id[prefix_id]] = PrefixPaths(_FullPaths(mapped, indexed, offsets_start+start*8, count))
        stream.seek(records*8, 1)
        global_ases = set(struct.unpack(f"<{num_asns}I", stream.read(num_asns*4)))
        global_edges = {(encoded >> 32, encoded & 0xffffffff)
                        for encoded in struct.unpack(f"<{num_edges}Q", stream.read(num_edges*8))}
    return {
        "entries": entries, "global_ases": global_ases, "global_edges": global_edges,
    }


def _text_stream(path):
    path = Path(path)
    opener = bz2.open if path.suffix == ".bz2" else gzip.open if path.suffix == ".gz" else open
    return opener(path, "rt", encoding="utf-8")


def load_ranks(path):
    ranks = {}
    with _text_stream(path) as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("rank") is not None and int(row["rank"]) > 0:
                ranks[int(row["asn"])] = int(row["rank"])
    return ranks


def load_relationships(path):
    relationships = {}
    with _text_stream(path) as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            left, right, rel, *_ = line.strip().split("|")
            left, right, rel = int(left), int(right), int(rel)
            relationships[left, right] = rel
            relationships[right, left] = -rel
    return relationships


def rank_tier(rank):
    if rank is None or rank < 1:
        return None
    for number, upper in enumerate((500, 2000, 5000, 15000), 1):
        if rank <= upper:
            return number
    return 5


def is_valley_free(path, relationships):
    phase = 0
    for left, right in zip(path, path[1:]):
        relation = relationships.get((left, right))
        if relation == 1:
            if phase:
                return False
        elif relation == 0:
            if phase:
                return False
            phase = 1
        elif relation == -1:
            phase = 2
        else:
            return False
    return True


def _candidate_pools(reference, ranks, relationships):
    tiers = defaultdict(list)
    for asn in sorted(reference["global_ases"]):
        tier = rank_tier(ranks.get(asn))
        if tier is not None:
            tiers[tier].append(asn)
    leaks = defaultdict(lambda: defaultdict(list))
    for au, ad in sorted(relationships):
        tier = rank_tier(ranks.get(ad))
        if (tier is not None and ad in reference["global_ases"]
                and relationships[ad, au] in (-1, 0)
                and (min(au, ad), max(au, ad)) in reference["global_edges"]):
            leaks[au][tier].append(ad)
    return tiers, leaks


def _record(scenario, prefix, path, depth, ad, ranks, sample_id):
    au, origin, level = path[-1-depth], path[-1], depth+2
    synthetic = (ad,) + path[-1-depth:]
    return dict(zip(MANIFEST_COLUMNS, [
        sample_id, scenario, prefix, au, ad, depth, ranks[ad], TIERS[rank_tier(ranks[ad])-1],
        " ".join(map(str, path)), " ".join(map(str, synthetic)), TIMESTAMP, ad, origin, level, level,
        int(ad == origin), 0, 0, str(au), "",
        {"path_manipulation": "new_prefix_link",
         "route_leak": "caida_valley_free_violation_new_prefix_link_existing_global_link"}[scenario],
    ]))


def _choose_ad(pool, eligible, rng):
    for _ in range(min(64, len(pool))):
        candidate = rng.choice(pool)
        if eligible(candidate):
            return candidate
    candidates = [candidate for candidate in pool if eligible(candidate)]
    return rng.choice(candidates) if candidates else None


class ExactCandidateHistory:
    def __init__(self, cache_dir, data_dir):
        cache_dir, data_dir = Path(cache_dir), Path(data_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        source = Path(__file__).with_name("candidate_history_index.cpp")
        binary = cache_dir / "candidate_history.so"
        if not binary.exists():
            subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC", "-mpopcnt",
                            str(source), "-o", str(binary)], check=True)
        self.library = ctypes.CDLL(str(binary.resolve()))
        pointer = ctypes.c_void_p
        self.library.ch_build.argtypes = [pointer, ctypes.c_uint64, pointer,
                                         ctypes.c_uint32, ctypes.c_uint32, ctypes.c_char_p]
        self.library.ch_build.restype = None
        self.library.ch_open.argtypes = [ctypes.c_char_p]
        self.library.ch_open.restype = pointer
        self.library.ch_close.argtypes = [pointer]
        self.library.ch_close.restype = None
        self.library.ch_eligible.argtypes = [pointer, ctypes.c_uint32, ctypes.c_uint32]
        self.library.ch_eligible.restype = ctypes.c_int
        index = cache_dir / "candidate_history.bin"
        if not index.exists():
            meta = json.loads((data_dir / "meta.json").read_text())
            triplets = torch.load(data_dir / "profile_triplets.pt", mmap=True, weights_only=True)
            fingerprints = torch.load(data_dir / "profile_fingerprints.pt", weights_only=True)
            self.library.ch_build(triplets.data_ptr(), len(triplets), fingerprints.data_ptr(),
                                  meta["num_profiles"], meta["num_asns"], str(index).encode())
        self.handle = self.library.ch_open(str(index).encode())

    def eligible(self, profile_id, as_id):
        return bool(self.library.ch_eligible(self.handle, int(profile_id), int(as_id)))

    def close(self):
        self.library.ch_close(self.handle)


class HistoricalCandidateRules:
    def __init__(self, cache_dir, data_dir):
        self.index = ExactCandidateHistory(cache_dir, data_dir)
        self.prefix_map = torch.load(data_dir / "prefix_map.pt", weights_only=True)
        self.prefix_to_profile = torch.load(data_dir / "prefix_to_profile.pt", weights_only=True)
        self.as_map = torch.load(data_dir / "as_map.pt", weights_only=True)

    def profile_id(self, prefix):
        return int(self.prefix_to_profile[self.prefix_map[prefix]])

    def eligible(self, profile_id, asn):
        return self.index.eligible(profile_id, self.as_map[asn])


def generate(reference, ranks, relationships, excluded, quotas, seed, history_rules):
    prefixes = sorted(set(reference["entries"]) - excluded)
    tiers, leaks = _candidate_pools(reference, ranks, relationships)
    cells = list(quotas)
    order = cells.copy()
    random.Random(seed).shuffle(order)
    used_prefixes, rows = set(), []
    counters, sampled = Counter(), Counter()

    for cell in order:
        scenario, depth, tier = cell
        rng = random.Random(seed * 1000 + cells.index(cell))
        prefix_order = prefixes.copy()
        rng.shuffle(prefix_order)
        for prefix in prefix_order:
            if prefix in used_prefixes:
                continue
            profile_id = history_rules.profile_id(prefix)
            all_paths = set(reference["entries"][prefix].paths)
            direct_history = {asn for path in all_paths for hop, asn in enumerate(reversed(path)) if hop < 8}
            prefix_edges = {(upstream, downstream) for path in all_paths
                            for downstream, upstream in zip(path, path[1:])}
            paths = sorted(path for path in all_paths if len(path) == len(set(path)))
            rng.shuffle(paths)
            chosen = None
            for path in paths:
                if len(path) <= depth:
                    continue
                au, suffix = path[-1-depth], path[-1-depth:]
                if scenario == "path_manipulation":
                    def eligible(ad):
                        return (ad not in direct_history and ad not in suffix and (au, ad) not in prefix_edges
                                and (min(au, ad), max(au, ad)) not in reference["global_edges"]
                                and (au, ad) not in relationships
                                and history_rules.eligible(profile_id, ad))
                    ad = _choose_ad(tiers.get(tier, []), eligible, rng)
                else:
                    if not is_valley_free(suffix, relationships):
                        continue
                    def eligible(ad):
                        return (ad not in direct_history and ad not in suffix and (au, ad) not in prefix_edges
                                and not is_valley_free((ad,) + suffix, relationships)
                                and history_rules.eligible(profile_id, ad))
                    ad = _choose_ad(leaks.get(au, {}).get(tier, []), eligible, rng)
                if ad is not None:
                    chosen = path, ad
                    break
            if chosen is not None:
                path, ad = chosen
                tag = "pm" if scenario == "path_manipulation" else "rl"
                row = _record(scenario, prefix, path, depth, ad, ranks,
                              f"{tag}_topology_{counters[scenario]:05d}")
                row["note"] = ("history_qualified_new_global_rib_and_caida_link" if tag == "pm"
                               else "history_qualified_known_valley_free_suffix_first_violation")
                rows.append(row)
                counters[scenario] += 1
                sampled[cell] += 1
                used_prefixes.add(prefix)
            if sampled[cell] == quotas[cell]:
                break
        print(f"{cell}: {sampled[cell]}/{quotas[cell]}", flush=True)
        if sampled[cell] != quotas[cell]:
            raise ValueError(f"Insufficient unused eligible prefixes for {cell}: "
                             f"{sampled[cell]}/{quotas[cell]}")
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS).sort_values(
        ["scenario", "depth", "rank", "sample_id"], kind="stable").reset_index(drop=True)


def plot_detection_heatmap(rows, output_dir):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                         "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
                         "xtick.labelsize": 11, "ytick.labelsize": 11, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    scenarios = ("path_manipulation", "route_leak")
    titles = ("Path Manipulation Detection Rate", "Route Leak Detection Rate")
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#d9d9d9")
    for ax, scenario, title in zip(axes, scenarios, titles):
        values = np.full((5, 5), np.nan)
        for row in rows:
            if row["scenario"] == scenario:
                values[TIERS.index(row["rank_tier"]), row["depth"]] = row["recall"]
        plot = ax.imshow(values, cmap=cmap, vmin=0, vmax=1, aspect="equal")
        for tier in range(5):
            for depth in range(5):
                value = values[tier, depth]
                label = "N/A" if np.isnan(value) else f"{value:.3f}"
                color = "#555555" if np.isnan(value) else "white" if value < .18 or value > .85 else "black"
                ax.text(depth, tier, label, ha="center", va="center", fontsize=10, color=color)
        ax.set(xticks=range(5), yticks=range(5), yticklabels=range(1, 6),
               xlabel="Propagation Depth", ylabel="Rank Tier", title=title)
        ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(plot, ax=axes, label="Detection Rate (Recall)", ticks=np.linspace(0, 1, 6), shrink=.96)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"diagnostic_heatmaps_paper.{extension}", dpi=300)
    plt.close(fig)


def summarize(frame, columns):
    return frame.groupby(columns)["prediction"].agg(samples="size", detected="sum", recall="mean").reset_index()


def generate_manifest(args, out, data, cache):
    history = HistoricalCandidateRules(args.candidate_history_cache or cache / "candidate_history", data)
    index = args.reference_index or cache / "reference_index.bin"
    reference = load_reference(index, reference_paths=data / "reference_paths.bin", prefix_map=history.prefix_map)
    rank_file = args.ranks or args.source / "as_info.2601.jsonl"
    excluded_file = args.excluded_mixed or args.source / "mixed_manifest.parquet"
    ranks = load_ranks(rank_file)
    relationships = load_relationships(args.relationships)
    excluded = set(pd.read_parquet(excluded_file, columns=["prefix"]).prefix)
    quotas = {(scenario, depth, tier): args.samples_per_cell
              for scenario, depths in (("path_manipulation", range(5)), ("route_leak", range(1, 5)))
              for depth in depths for tier in range(1, 6)}
    config = {"data_dir": str(data), "reference_paths": str(data / "reference_paths.bin"),
              "reference_index": str(index), "ranks": str(rank_file),
              "relationships": str(args.relationships), "excluded_mixed": str(excluded_file),
              "seed": args.seed, "samples_per_cell": args.samples_per_cell,
              "sampling": {"direct_hop_exclusive": 8, "similar_hop_exclusive": 4,
                           "fingerprint_similarity_strictly_above": 0.75,
                           "prefixes_globally_unique": True, "paths_per_prefix": "all"}}
    (out / "generation_config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = generate(reference, ranks, relationships, excluded, quotas, args.seed, history)
    manifest.to_parquet(out / "diagnostic_manifest.parquet", index=False)
    history.index.close()


def score_manifest(out, data, model):
    frame = pd.read_parquet(out / "diagnostic_manifest.parquet")
    resources = load_resources(data, model, 24, 2)
    thresholds = load_thresholds(data, model)
    scored = score_candidates(frame, resources, thresholds, anchor_for_all=True, batch_size=8192)
    scored.to_parquet(out / "diagnostic_scores.parquet", index=False)
    cells = summarize(scored, ["scenario", "rank_tier", "depth"])
    cells.to_csv(out / "diagnostic_recall.csv", index=False)
    plot_detection_heatmap(cells.to_dict("records"), out)
    summary = {"rows": len(scored), "device": str(resources[4]), "thresholds": list(thresholds),
               "policy": "geometry + fingerprint", "data_dir": str(data), "model_dir": str(model),
               "by_scenario": summarize(scored, ["scenario"]).to_dict("records")}
    (out / "score_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--ranks", type=Path)
    parser.add_argument("--excluded-mixed", type=Path)
    parser.add_argument("--relationships", type=Path, default=Path("/networkDisk/misc/caida_asrel/20260101.as-rel.txt"))
    parser.add_argument("--reference-index", type=Path)
    parser.add_argument("--candidate-history-cache", type=Path)
    parser.add_argument("--samples-per-cell", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    out = args.output_dir or ROOT / "experiments/results/topology"
    cache = ROOT / "experiments/cache" / out.name
    data = args.data_dir or args.source / "coverage/c100/data"
    model = args.model_dir or args.source / "coverage/c100/model"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    if not (out / "diagnostic_manifest.parquet").exists():
        generate_manifest(args, out, data, cache)
    score_manifest(out, data, model)
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
