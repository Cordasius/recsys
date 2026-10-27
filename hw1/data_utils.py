"""Amazon 2023: selected 5-core departments, shared IDs and train-only sampling."""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import sparse

DATASET = "McAuley-Lab/Amazon-Reviews-2023"
REVISION = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"
CATEGORIES = ("Beauty_and_Personal_Care", "Arts_Crafts_and_Sewing",
              "Musical_Instruments", "Industrial_and_Scientific")
DATA_DIRECTORY = "data/amazon_5core"
SEED = 42
MAX_ITEMS = 6000
MAX_USERS = 30000
METADATA_SHARDS = {"Arts_Crafts_and_Sewing": 4, "Musical_Instruments": 2,
                   "Industrial_and_Scientific": 2}
RATING_FILES = {domain: f"benchmark/5core/rating_only/{domain}.csv" for domain in CATEGORIES}
METADATA_FILES = {
    domain: ([f"raw_meta_{domain}/full-{i:05d}-of-{METADATA_SHARDS[domain]:05d}.parquet"
              for i in range(METADATA_SHARDS[domain])]
             if domain in METADATA_SHARDS else [f"raw/meta_categories/meta_{domain}.jsonl"])
    for domain in CATEGORIES
}
FILES = tuple(RATING_FILES.values()) + tuple(p for paths in METADATA_FILES.values() for p in paths)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_data_ready(directory=DATA_DIRECTORY):
    root = Path(directory)
    manifest = root / "source_manifest.json"
    if not manifest.is_file():
        return False
    info = json.loads(manifest.read_text())
    return (info.get("revision") == REVISION and set(info.get("files", {})) == set(FILES)
            and all((root / name).is_file() and (root / name).stat().st_size == spec["size"]
                    for name, spec in info["files"].items()))


def download_plan(token=False, max_gb=5.0):
    """Query sizes only; no data bodies are downloaded by this function."""
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    rows = []
    for filename in FILES:
        url = hf_hub_url(DATASET, filename, repo_type="dataset", revision=REVISION)
        info = get_hf_file_metadata(url, token=token)
        if info.size is None:
            raise ValueError(f"Не удалось определить размер {filename}; скачивание не начато")
        rows.append({"file": filename, "bytes": int(info.size), "GB": info.size / 1e9,
                     "etag": info.etag})
    plan = pd.DataFrame(rows)
    if plan.bytes.sum() > max_gb * 1e9:
        raise ValueError(f"Файлы занимают {plan.GB.sum():.3f} ГБ, больше лимита {max_gb}. "
                         "Данные не скачивались.")
    return plan


def download_data(plan, token=False, directory=DATA_DIRECTORY):
    """Download exactly the validated files through HF; token is never persisted."""
    from huggingface_hub import hf_hub_download
    if set(plan.file) != set(FILES) or plan.bytes.sum() > 5_000_000_000:
        raise ValueError("Unexpected download plan")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    records = {}
    for record in plan.to_dict("records"):
        name = record["file"]
        path = Path(hf_hub_download(DATASET, name, repo_type="dataset", revision=REVISION,
                                    token=token, local_dir=root))
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"Неполный файл: {name}")
        checksum = file_hash(path)
        
        if record["etag"] and len(record["etag"]) == 64 and checksum != record["etag"]:
            raise ValueError(f"SHA256 не совпадает: {name}")
        records[name] = {"size": path.stat().st_size, "sha256": checksum}
    manifest = {"dataset": DATASET, "revision": REVISION, "source_core": 5,
                "categories": list(CATEGORIES), "files": records,
                "total_source_bytes": int(plan.bytes.sum())}
    (root / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    return root


def load_reviews(directory=DATA_DIRECTORY):
    """CSV parser from datasets; Arrow-backed strings keep memory use manageable."""
    from datasets import Features, Value, load_dataset
    root = Path(directory)
    missing = [name for name in FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Сначала выполните ячейку скачивания в ноутбуке: " + ", ".join(missing))
    frames = []
    features = Features({"user_id": Value("string"), "parent_asin": Value("string"),
                         "rating": Value("float32"), "timestamp": Value("int64")})
    for domain, filename in RATING_FILES.items():
        dataset = load_dataset("csv", data_files=str(root / filename), split="train",
                               features=features, cache_dir=str(root / "arrow"))
        frame = dataset.data.table.to_pandas(types_mapper=pd.ArrowDtype)
        frame["domain"] = domain
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    required = ["user_id", "parent_asin", "rating", "timestamp", "domain"]
    audit = {"source_rows": len(raw), "source_users": raw.user_id.nunique(),
             "source_items": raw.parent_asin.nunique(), "source_domains": raw.domain.nunique(),
             "missing": raw[required].isna().sum().to_dict(),
             "exact_duplicates": int(raw.duplicated().sum()),
             "pair_duplicates": int(raw.duplicated(["user_id", "parent_asin"]).sum())}
    clean = raw[required].copy()
    for column in ["rating", "timestamp"]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    valid = (clean[required].notna().all(axis=1) & clean.rating.between(1, 5)
             & clean.timestamp.between(0, 1_700_000_000_000)
             & clean.user_id.str.len().gt(0) & clean.parent_asin.str.len().gt(0))
    clean = clean.loc[valid].sort_values("timestamp", kind="stable")
    clean = clean.drop_duplicates(["user_id", "parent_asin"], keep="first")

    canonical = clean.drop_duplicates("parent_asin").set_index("parent_asin").domain
    clean["domain"] = clean.parent_asin.map(canonical)
    clean["timestamp"] = clean.timestamp.astype("int64")
    cross = clean.groupby("user_id").domain.nunique()
    if not (cross >= 2).any():
        raise ValueError("В исходных ID нет общих пользователей между категориями")
    audit.update(clean_rows=len(clean), dropped_rows=len(raw) - len(clean),
                 users_in_multiple_domains=int((cross >= 2).sum()))
    return raw, clean.reset_index(drop=True), audit


def temporal_cutoffs(clean):
    """Common timestamp-only quantiles, before catalog/user selection."""
    train_end, valid_end = map(int, clean.timestamp.quantile(
        [.8, .9], interpolation="higher").to_numpy())
    if not clean.timestamp.min() < train_end < valid_end <= clean.timestamp.max():
        raise ValueError("Insufficient distinct timestamps for temporal splitting")
    return train_end, valid_end


def prepare_sample(clean, train_end, valid_end, seed=SEED, max_items=MAX_ITEMS, max_users=MAX_USERS,
                   activity_days=730, min_user_items=3, domains=CATEGORIES):
    if not train_end < valid_end:
        raise ValueError("train_end must be earlier than valid_end")
    train_all = clean[clean.timestamp < train_end]
    positive = train_all[train_all.rating >= 4]
    domains = sorted(domains)
    quota, remainder = divmod(max_items, len(domains))
    items, active_users = set(), set()
    for index, domain in enumerate(domains):
        history = positive[positive.domain == domain]
        recent = history[history.timestamp >= train_end - activity_days * 86_400_000]
        focus = recent if not recent.empty else history
        counts = focus.groupby("parent_asin").user_id.nunique().rename("count").reset_index()
        counts = counts[counts["count"] >= 2].sort_values(
            ["count", "parent_asin"], ascending=[False, True], kind="stable")
        chosen = counts.head(quota + (index < remainder)).parent_asin
        if chosen.empty:
            raise ValueError(f"Нет обучающих товаров с поддержкой >=2 в {domain}")
        items.update(chosen)
        active_users.update(focus[focus.parent_asin.isin(chosen)].user_id)
    positive = positive[positive.parent_asin.isin(items)]
    user_counts = positive.groupby("user_id").parent_asin.nunique()
    eligible = np.sort(user_counts[(user_counts >= min_user_items)
                                    & user_counts.index.isin(active_users)].index.to_numpy())
    rng = np.random.default_rng(seed)
    reserved = set()
    reserve_per_domain = max(1, max_users // (2 * len(domains)))
    for domain in domains:
        candidates = np.sort(positive[(positive.domain == domain)
                                       & positive.user_id.isin(eligible)].user_id.unique())
        if not len(candidates):
            raise ValueError(f"Нет подходящих обучающих пользователей для {domain}")
        reserved.update(rng.choice(candidates, min(reserve_per_domain, len(candidates)), replace=False))
    remaining = np.array([u for u in eligible if u not in reserved])
    additional = rng.choice(remaining, min(max_users - len(reserved), len(remaining)), replace=False)
    users = np.array(sorted(reserved | set(additional)))
    selected_train = positive[positive.user_id.isin(users)]
    items = np.sort(selected_train.parent_asin.unique())
    if set(selected_train.domain) != set(domains):
        raise ValueError("Категории потеряны при отборе аудитории")
    sample = clean[clean.user_id.isin(users) & clean.parent_asin.isin(items)].copy()
    if not (100_000 <= len(sample) <= 2_000_000 and len(users) >= 1000 and len(items) >= 1000):
        raise ValueError(f"Выборка не соответствует условию: событий {len(sample)}, "
                         f"пользователей {len(users)}, товаров {len(items)}. "
                         "Измените обучающий бюджет до подбора моделей.")
    
    sample["user_idx"] = pd.Index(users).get_indexer(sample.user_id)
    sample["item_idx"] = pd.Index(items).get_indexer(sample.parent_asin)
    train = sample[sample.timestamp < train_end].copy()
    valid = sample[sample.timestamp.between(train_end, valid_end, inclusive="left")].copy()
    test = sample[sample.timestamp >= valid_end].copy()
    if not (train.groupby("user_id").domain.nunique() >= 2).any():
        raise ValueError("После отбора не осталось пользователей с несколькими категориями")
    assert train.timestamp.max() < train_end
    if len(valid):
        assert valid.timestamp.min() >= train_end and valid.timestamp.max() < valid_end
    if len(test):
        assert test.timestamp.min() >= valid_end
    return sample, train, valid, test, users, items, (len(users), len(items))


def load_metadata(items, directory=DATA_DIRECTORY, item_domains=None):
    import pyarrow.parquet as pq
    root = Path(directory)
    wanted, records, total = set(items), {}, 0
    columns = ["parent_asin", "title", "store", "categories"]

    def consume(obj, domain):
        nonlocal total
        total += 1
        item = obj.get("parent_asin")
        if item not in wanted or item in records:
            return
        if item_domains is not None and item_domains[item] != domain:
            return
        categories = obj.get("categories") or []
        if not isinstance(categories, list):
            categories = [str(categories)]
        path = [str(c).strip() for c in categories if c]
        records[item] = {"parent_asin": item, "title": obj.get("title") or "",
                         "store": obj.get("store") or "<missing>", "domain": domain,
                         "category": " > ".join([domain] + (path or ["<missing>"]))}

    for domain, filenames in METADATA_FILES.items():
        for filename in filenames:
            if filename.endswith(".parquet"):
                for batch in pq.ParquetFile(root / filename).iter_batches(batch_size=8192, columns=columns):
                    for obj in batch.to_pylist():
                        consume(obj, domain)
            else:
                with (root / filename).open(encoding="utf-8") as stream:
                    for line in stream:
                        consume(json.loads(line), domain)
    
    for item in wanted - set(records):
        domain = item_domains[item] if item_domains is not None else "<missing>"
        records[item] = {"parent_asin": item, "title": "", "store": "<missing>",
                         "domain": domain, "category": domain + " > <missing>"}
    metadata = pd.DataFrame([records[item] for item in items])
    codes = []
    for column in ["domain", "category", "store"]:
        values = metadata[column]
        if column != "domain":
            frequency = values.value_counts()
            values = values.where(values.map(frequency) >= 5, "<rare>")
        code, _ = pd.factorize(values, sort=True)
        codes.append(code)
    return metadata, np.stack(codes, axis=1), total


def interaction_matrix(frame, shape, positive_only=True):
    if positive_only:
        frame = frame[frame.rating >= 4]
    matrix = sparse.csr_matrix((np.ones(len(frame)), (frame.user_idx, frame.item_idx)), shape=shape)
    matrix.data[:] = 1.0
    return matrix
