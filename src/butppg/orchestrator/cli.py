"""``orch`` — the BUT PPG orchestrator CLI.

Three independent commands, run separately (not a single pipeline):

    orch mart     ...   # BUT PPG -> per-model input marts (SIGMA-PPG / OpenTSLM)
    orch train    ...   # launch training of one model on one task -> a run
    orch results  ...   # inspect a run's metrics / download its trained weights

Each command stands alone: ``mart`` produces reusable marts, ``train`` consumes
them (or the registry directly) and writes a self-contained ``runs/<id>/``, and
``results`` reads those run folders. Designed so the same CLI works on a laptop
(marts + trivial/CNN baselines) and on the GPU box (SIGMA-PPG / OpenTSLM).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from butppg.paths import MARTS_DIR, PROJECT_ROOT, RESULTS_DIR, SPLITS_DIR

# Short aliases for --llm-id so you don't type full HF repo ids. A value that is
# not an alias is passed through verbatim (any HF repo id still works).
LLM_ALIASES = {
    "llama-3b": "meta-llama/Llama-3.2-3B",   # assignment default (OpenTSLM-Flamingo)
    "llama-1b": "meta-llama/Llama-3.2-1B",   # OpenTSLM's own default
    "gemma-1b": "google/gemma-3-1b-pt",      # light debug model (base/pretrained)
    "gemma-270m": "google/gemma-3-270m",     # smallest — fastest pipeline smoke test
}


def resolve_llm_id(value: str) -> str:
    """Map a short alias to its HF repo id; pass through anything else unchanged."""
    return LLM_ALIASES.get(value, value)


# --------------------------------------------------------------------------- #
# mart                                                                        #
# --------------------------------------------------------------------------- #
def _default_registry() -> str:
    return str(PROJECT_ROOT / "data" / "processed" / "registry.csv")


def _default_split() -> str:
    return str(PROJECT_ROOT / "splits" / "but_ppg_60_20_20.json")


def cmd_mart(args: argparse.Namespace) -> int:
    from butppg.data.marts_build import build_opentslm_mart, build_sigma_mart

    registry = args.registry
    split = args.split

    if args.prepare:
        from butppg.data.prepare import prepare_but_ppg

        print(f"[mart] preparing BUT PPG (limit={args.limit}) ...")
        registry = str(prepare_but_ppg(out_dir=Path(registry).parent, limit=args.limit, workers=args.workers))
        print(f"[mart] registry -> {registry}")

    if args.make_split or not Path(split).exists():
        _build_split(registry, split, args.seed)

    models = ["sigma_ppg", "opentslm"] if args.model == "both" else [args.model]
    tasks = ["quality", "hr"] if args.task == "both" else [args.task]

    for model in models:
        for task in tasks:
            out_root = Path(args.out) / model
            if model == "sigma_ppg":
                m = build_sigma_mart(
                    registry, split, out_root, task,
                    target_fs=args.target_fs, normalize=args.normalize, seed=args.seed,
                )
                counts = {k: v["windows"] for k, v in m["splits"].items()}
            else:
                m = build_opentslm_mart(
                    registry, split, out_root, task, acc_mode=args.acc_mode, seed=args.seed,
                )
                counts = {k: v["records"] for k, v in m["splits"].items()}
            print(f"[mart] {model}/{task}: {counts} -> {out_root / task}")
    return 0


def _build_split(registry_path: str, split_path: str, seed: int) -> None:
    from butppg.data.registry import load_registry
    from butppg.data.splits import make_subject_split, save_split

    reg_df = load_registry(registry_path)
    tr, va, te = make_subject_split(reg_df, seed=seed)
    save_split(tr, va, te, split_path, seed=seed)
    print(f"[split] {split_path}  (train {len(tr)} / val {len(va)} / test {len(te)} subjects)")


# --------------------------------------------------------------------------- #
# ingest  (RAW layer)                                                         #
# --------------------------------------------------------------------------- #
def cmd_ingest(args: argparse.Namespace) -> int:
    from butppg.data.raw import ingest_raw

    ingest_raw(
        raw_dir=args.raw_dir,
        limit=args.limit,
        include_acc=not args.no_acc,
        skip_existing=not args.no_skip_existing,
        workers=args.workers,
    )
    return 0


# --------------------------------------------------------------------------- #
# process  (raw -> processed windows + registry + split)                      #
# --------------------------------------------------------------------------- #
def cmd_process(args: argparse.Namespace) -> int:
    from butppg.data.process import process_records

    registry_path = process_records(raw_dir=args.raw_dir, out_dir=args.out_dir)
    if args.make_split or not Path(args.split).exists():
        _build_split(str(registry_path), args.split, args.seed)
    return 0


# --------------------------------------------------------------------------- #
# train                                                                       #
# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace) -> int:
    from butppg.orchestrator.runs import create_run
    from butppg.training.dispatch import run_training

    cfg: dict = {
        "registry": args.registry,
        "split": args.split,
        "device": args.device,
        "train": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr},
    }
    if args.model in ("sigma_ppg", "opentslm"):
        cfg["mart_dir"] = args.mart_dir or str(MARTS_DIR / args.model)
    if args.model == "sigma_ppg":
        cfg["checkpoint_path"] = args.checkpoint_path
        cfg["target_fs"] = args.target_fs
        if args.patch_size:
            cfg["patch_size"] = args.patch_size
    if args.model == "opentslm":
        cfg["llm_id"] = resolve_llm_id(args.llm_id)
        cfg["ecg_init"] = args.ecg_init
    if args.model == "cnn1d":
        channels = ["ppg", "acc"] if args.input_variant in ("ppg_acc", "ppg_acc_cov") else ["ppg"]
        cfg["model"] = {"arch": args.arch, "channels": channels}
        cfg["input_variant"] = args.input_variant
    if args.model == "baseline_features":
        cfg["input_variant"] = args.input_variant
        cfg["estimator"] = args.estimator
    if args.model == "trivial":
        cfg["method"] = args.method  # hr only: 'median' | 'dominant_frequency'

    run = create_run(args.model, args.task, config=cfg)
    print(f"[train] run_id = {run.run_id}")
    print(f"[train] model={args.model} task={args.task} device={args.device}")
    try:
        run_training(run, cfg)
    except Exception as exc:  # dispatch already recorded status=failed
        print(f"[train] FAILED: {type(exc).__name__}: {exc}")
        print(f"[train] see {run.log_file}")
        return 1

    run = _reload(run.run_id)
    print(f"[train] status={run.status}")
    _print_metrics(run.metrics)
    print(f"[train] run dir: {run.dir}")
    return 0


def _reload(run_id: str):
    from butppg.orchestrator.runs import load_run

    return load_run(run_id)


# --------------------------------------------------------------------------- #
# results                                                                     #
# --------------------------------------------------------------------------- #
def cmd_results(args: argparse.Namespace) -> int:
    from butppg.orchestrator.runs import list_runs, load_run

    if not args.run_id:
        runs = list_runs()
        if not runs:
            print("no runs yet. Launch one with `orch train ...`")
            return 0
        print(f"{'RUN_ID':<44} {'MODEL':<12} {'TASK':<8} {'STATUS':<10} PRIMARY")
        for r in runs:
            print(f"{r.run_id:<44} {r.model:<12} {r.task:<8} {r.status:<10} {_primary(r)}")
        return 0

    run = load_run(args.run_id)
    print(f"run_id:   {run.run_id}")
    print(f"model:    {run.model}    task: {run.task}    status: {run.status}")
    print(f"created:  {run.created_at}")
    if run.error:
        print(f"error:    {run.error}")
    print(f"run dir:  {run.dir}")
    if run.best_checkpoint:
        print(f"weights:  {run.best_checkpoint}")
    _print_metrics(run.metrics)

    if args.download:
        return _download_weights(run, Path(args.download))
    return 0


def _download_weights(run, out_dir: Path) -> int:
    if run.status != "finished":
        print(f"[results] run status is {run.status!r}; weights may be missing/partial")
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    if run.best_checkpoint and Path(run.best_checkpoint).exists():
        dst = out_dir / Path(run.best_checkpoint).name
        shutil.copy2(run.best_checkpoint, dst)
        copied.append(dst)
    # ship metrics + predictions alongside the weights so the export is usable on its own
    for sub in ("metrics", "predictions"):
        src = run.dir / sub
        if src.exists():
            shutil.copytree(src, out_dir / sub, dirs_exist_ok=True)
            copied.append(out_dir / sub)
    if not copied:
        print("[results] nothing to download (no checkpoint found for this run)")
        return 1
    print(f"[results] exported to {out_dir}:")
    for c in copied:
        print(f"  {c}")
    return 0


# --------------------------------------------------------------------------- #
# cascade  (quality-gate -> HR scenario, section 2B)                          #
# --------------------------------------------------------------------------- #
def cmd_cascade(args: argparse.Namespace) -> int:
    import json

    from butppg.evaluation.evaluate import cascade_report
    from butppg.orchestrator.runs import load_run

    q_run = load_run(args.quality_run)
    q_path = q_run.predictions_dir / "test_predictions.csv"
    h_path = None
    if args.hr_run:
        h_path = load_run(args.hr_run).predictions_dir / "test_predictions.csv"

    report = cascade_report(q_path, h_path)
    print("[cascade] quality-gate -> HR (section 2B)")
    print(f"  accepted fraction : {report['accepted_fraction']:.4f}  ({report['n_accepted']}/{report['n_test']})")
    print(f"  false-reject rate : {_fmtopt(report['false_reject_rate'])}  (good windows wrongly rejected)")
    print(f"  false-accept rate : {_fmtopt(report['false_accept_rate'])}  (bad windows wrongly accepted)")
    if report.get("hr_metrics_on_accepted"):
        m = report["hr_metrics_on_accepted"]
        print(f"  HR on accepted    : MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  (n={report['hr_windows_scored']})")

    out = RESULTS_DIR / "tables" / f"cascade_{args.quality_run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[cascade] report -> {out}")
    return 0


# --------------------------------------------------------------------------- #
# table  (assemble comparison tables, section 10)                             #
# --------------------------------------------------------------------------- #
def cmd_table(args: argparse.Namespace) -> int:
    from butppg.reporting.tables import write_comparison

    paths = write_comparison()
    print("[table] wrote:")
    for _, p in paths.items():
        print(f"  {p}")
    print()
    print((paths["main_md"]).read_text(encoding="utf-8"))
    return 0


# --------------------------------------------------------------------------- #
# predict  (run a trained model on new data from its checkpoint, section 8)   #
# --------------------------------------------------------------------------- #
def cmd_predict(args: argparse.Namespace) -> int:
    from butppg.inference.predict import predict_run

    run = predict_run(
        args.run, registry=args.registry, split_path=args.split_path,
        split=args.split, mart_dir=args.mart_dir, tag=args.tag,
    )
    print(f"[predict] {run.run_id}  status={run.status}  (weights from {args.run})")
    _print_metrics(run.metrics or {})
    print(f"[predict] run dir: {run.dir}")
    return 0


# --------------------------------------------------------------------------- #
# galaxy  (build GalaxyPPG registry for external validation, section 8)       #
# --------------------------------------------------------------------------- #
def cmd_galaxy(args: argparse.Namespace) -> int:
    import pandas as pd

    from butppg.data.galaxy import build_galaxy_registry
    from butppg.data.splits import save_split

    reg_path = build_galaxy_registry(args.raw_dir, args.out, invert_ppg=not args.no_invert)
    reg = pd.read_csv(reg_path, dtype={"subject_id": str})
    subs = sorted(reg["subject_id"].unique())
    split_path = Path(args.out) / "galaxy_all_test.json"
    save_split([], [], subs, split_path, seed=42, ratios=(0.0, 0.0, 1.0))
    print(f"[galaxy] all-test split -> {split_path}  ({len(subs)} subjects, all test)")
    return 0


# --------------------------------------------------------------------------- #
# activity-report  (GalaxyPPG HR stratified by activity, section 8)           #
# --------------------------------------------------------------------------- #
def cmd_activity_report(args: argparse.Namespace) -> int:
    import json

    from butppg.evaluation.evaluate import activity_report
    from butppg.orchestrator.runs import load_run
    from butppg.reporting.galaxy_table import write_galaxy_table

    hr_path = load_run(args.hr_run).predictions_dir / "test_predictions.csv"
    q_path = load_run(args.quality_run).predictions_dir / "test_predictions.csv" if args.quality_run else None
    rep = activity_report(hr_path, args.registry, q_path)

    print("[galaxy] external HR validation")
    ov = rep.get("overall") or {}
    print(f"  overall: MAE={_fmtopt(ov.get('mae'))}  RMSE={_fmtopt(ov.get('rmse'))}  (n={rep['n_scored']})")
    for act, m in sorted(rep["per_activity"].items()):
        print(f"    {act:<16} MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  (n={m['n']})")
    if "accepted_fraction" in rep:
        print(f"  quality-accepted fraction: {_fmtopt(rep['accepted_fraction'])}")

    out = RESULTS_DIR / "tables" / f"galaxy_{args.hr_run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    md = write_galaxy_table(rep, RESULTS_DIR / "tables" / "galaxy_activity.md")
    print(f"[galaxy] report -> {out}\n[galaxy] table -> {md}")
    return 0


# --------------------------------------------------------------------------- #
# lowdata-subsets  (nested train subsets for the low-data study, section 9)   #
# --------------------------------------------------------------------------- #
def cmd_lowdata_subsets(args: argparse.Namespace) -> int:
    import json

    from butppg.data.splits import load_split, make_low_data_subsets, save_split

    base = load_split(args.split)
    fracs = [float(x) for x in args.fractions.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    made = []
    for s in seeds:
        subsets = make_low_data_subsets(base["train_subjects"], fracs, seed=s)
        for f, train_subs in subsets.items():
            path = outdir / f"f{int(round(f * 100))}_s{s}.json"
            save_split(train_subs, base["val_subjects"], base["test_subjects"], path,
                       seed=s, ratios=base.get("ratios", (0.6, 0.2, 0.2)))
            d = json.loads(path.read_text(encoding="utf-8"))
            d["low_data"] = {"fraction": f, "seed": s, "n_train_subjects": len(train_subs)}
            path.write_text(json.dumps(d, indent=2), encoding="utf-8")
            made.append(path)
    print(f"[lowdata] wrote {len(made)} split files to {outdir}/ (fractions={fracs}, seeds={seeds})")
    print("  train each model per split, e.g.:")
    print(f"    orch train --model baseline_features --task quality --split {made[0]}")
    return 0


# --------------------------------------------------------------------------- #
# lowdata-table  (aggregate low-data runs -> curves + tables, section 9/10)   #
# --------------------------------------------------------------------------- #
def cmd_lowdata_table(args: argparse.Namespace) -> int:
    from butppg.reporting.lowdata import write_low_data_report

    paths = write_low_data_report()
    print("[lowdata] wrote:")
    for _, p in paths.items():
        print(f"  {p}")
    return 0


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _fmtopt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"


def _primary(run) -> str:
    m = run.metrics or {}
    if run.task == "quality" and "macro_f1" in m:
        return f"macro_f1={m['macro_f1']:.4f}"
    if run.task == "hr" and "mae" in m:
        return f"MAE={m['mae']:.3f}"
    return "-"


def _print_metrics(metrics: dict) -> None:
    if not metrics:
        return
    keys = ("macro_f1", "roc_auc", "accuracy") if "macro_f1" in metrics else ("mae", "rmse")
    parts = [f"{k}={metrics[k]:.4f}" for k in keys if metrics.get(k) is not None]
    if parts:
        print("[metrics] " + "  ".join(parts))


# --------------------------------------------------------------------------- #
# parser                                                                      #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    # mart
    m = sub.add_parser("mart", help="build per-model input marts from BUT PPG")
    m.add_argument("--model", choices=["sigma_ppg", "opentslm", "both"], default="both")
    m.add_argument("--task", choices=["quality", "hr", "both"], default="both")
    m.add_argument("--registry", default=_default_registry())
    m.add_argument("--split", default=_default_split())
    m.add_argument("--out", default=str(MARTS_DIR))
    m.add_argument("--prepare", action="store_true", help="download+preprocess BUT PPG first")
    m.add_argument("--limit", type=int, default=None, help="with --prepare: only N records (smoke test)")
    m.add_argument("--make-split", action="store_true", help="(re)build the subject-wise split")
    m.add_argument("--workers", type=int, default=8, help="with --prepare: parallel download threads")
    m.add_argument("--target-fs", type=float, default=50.0, help="SIGMA-PPG resample rate")
    m.add_argument("--normalize", choices=["zscore", "minmax"], default="zscore")
    m.add_argument("--acc-mode", choices=["none", "magnitude", "axes"], default="magnitude", help="OpenTSLM ACC")
    m.add_argument("--seed", type=int, default=42)
    m.set_defaults(func=cmd_mart)

    raw_default = str(PROJECT_ROOT / "data" / "raw")
    processed_default = str(PROJECT_ROOT / "data" / "processed")

    # ingest (RAW layer)
    ing = sub.add_parser("ingest", help="download raw BUT PPG signals to data/raw (slow, cached)")
    ing.add_argument("--raw-dir", default=raw_default)
    ing.add_argument("--limit", type=int, default=None, help="only the first N records (smoke test)")
    ing.add_argument("--no-acc", action="store_true", help="skip accelerometer download")
    ing.add_argument("--no-skip-existing", action="store_true", help="re-download even if already cached")
    ing.add_argument("--workers", type=int, default=8, help="parallel download threads (latency-bound; try 16)")
    ing.set_defaults(func=cmd_ingest)

    # process (raw -> processed + registry + split)
    pr = sub.add_parser("process", help="build registry + processed windows + split from data/raw (fast, local)")
    pr.add_argument("--raw-dir", default=raw_default)
    pr.add_argument("--out-dir", default=processed_default)
    pr.add_argument("--split", default=_default_split())
    pr.add_argument("--make-split", action="store_true", help="(re)build the subject-wise split")
    pr.add_argument("--seed", type=int, default=42)
    pr.set_defaults(func=cmd_process)

    # train
    t = sub.add_parser("train", help="launch training of one model on one task")
    t.add_argument("--model", required=True,
                   choices=["trivial", "cnn1d", "baseline_features", "sigma_ppg", "opentslm"])
    t.add_argument("--task", required=True, choices=["quality", "hr"])
    t.add_argument("--registry", default=_default_registry())
    t.add_argument("--split", default=_default_split())
    t.add_argument("--mart-dir", default=None, help="marts dir (sigma_ppg/opentslm); default artifacts/data_marts/<model>")
    t.add_argument("--device", default="cpu")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--arch", default="resnet1d", help="cnn1d: cnn1d|resnet1d")
    t.add_argument("--method", default="median", choices=["median", "dominant_frequency"],
                   help="trivial hr baseline: median HR or dominant-frequency HR")
    t.add_argument("--input-variant", default="ppg", choices=["ppg", "ppg_acc", "ppg_acc_cov"],
                   help="input blocks for baseline_features/cnn1d (section 2A)")
    t.add_argument("--estimator", default="logreg", choices=["logreg", "ridge", "xgboost"],
                   help="baseline_features estimator (quality: logreg/xgboost; hr: ridge/xgboost)")
    t.add_argument("--checkpoint-path", default=None, help="sigma_ppg: pretrained SIGMA checkpoint")
    t.add_argument("--target-fs", type=float, default=50.0, help="sigma_ppg: mart resample rate")
    t.add_argument("--patch-size", type=int, default=None, help="sigma_ppg: patch size (default=target_fs)")
    t.add_argument(
        "--llm-id", default="llama-3b",
        help="opentslm: base LLM. Aliases: llama-3b, llama-1b, gemma-1b, gemma-270m "
             "(or any HF repo id verbatim).",
    )
    t.add_argument("--ecg-init", default=None, help="opentslm: ECG-stage checkpoint for ECG->PPG transfer")
    t.set_defaults(func=cmd_train)

    # results
    r = sub.add_parser("results", help="inspect runs / download trained weights")
    r.add_argument("run_id", nargs="?", default=None, help="omit to list all runs")
    r.add_argument("--download", default=None, metavar="DIR", help="export weights (+metrics+predictions) to DIR")
    r.set_defaults(func=cmd_results)

    # cascade
    c = sub.add_parser("cascade", help="quality-gate -> HR scenario metrics (section 2B)")
    c.add_argument("--quality-run", required=True, help="run_id of a finished quality run (full test set)")
    c.add_argument("--hr-run", default=None, help="optional run_id of an HR run (HR MAE on accepted windows)")
    c.set_defaults(func=cmd_cascade)

    # table
    tb = sub.add_parser("table", help="assemble the section-10 comparison tables from finished runs")
    tb.set_defaults(func=cmd_table)

    # predict (inference from a trained checkpoint, section 8)
    pd_ = sub.add_parser("predict", help="run a trained model on new data from its checkpoint (no retraining)")
    pd_.add_argument("--run", required=True, help="run_id of a finished run whose weights to load")
    pd_.add_argument("--registry", default=None, help="target registry (registry models); default = training registry")
    pd_.add_argument("--split-path", default=None, help="split file to slice the registry (omit for whole registry)")
    pd_.add_argument("--split", default="test", help="which fold when --split-path given (default test; 'all')")
    pd_.add_argument("--mart-dir", default=None, help="target mart dir (sigma_ppg/opentslm); default = training mart")
    pd_.add_argument("--tag", default="predict", help="label for the output run id")
    pd_.set_defaults(func=cmd_predict)

    # galaxy (build GalaxyPPG registry, section 8)
    gx = sub.add_parser("galaxy", help="build GalaxyPPG registry + all-test split for external validation")
    gx.add_argument("--raw-dir", required=True, help="extracted GalaxyPPG release dir (Zenodo 10.5281/zenodo.14635823)")
    gx.add_argument("--out", default=str(PROJECT_ROOT / "data" / "galaxy"))
    gx.add_argument("--no-invert", action="store_true", help="do NOT invert PPG (default inverts: reflective sensor)")
    gx.set_defaults(func=cmd_galaxy)

    # activity-report (GalaxyPPG HR by activity, section 8)
    ar = sub.add_parser("activity-report", help="GalaxyPPG HR MAE overall + per activity (section 8)")
    ar.add_argument("--hr-run", required=True, help="run_id of an HR predict-run on GalaxyPPG")
    ar.add_argument("--registry", required=True, help="GalaxyPPG registry.csv (has the activity column)")
    ar.add_argument("--quality-run", default=None, help="optional quality predict-run on GalaxyPPG (accept fraction)")
    ar.set_defaults(func=cmd_activity_report)

    # lowdata-subsets (nested train subsets, section 9)
    ls = sub.add_parser("lowdata-subsets", help="generate nested train subsets (val/test fixed) for the low-data study")
    ls.add_argument("--split", default=_default_split(), help="base split to shrink the train fold of")
    ls.add_argument("--fractions", default="0.25,0.5,1.0", help="comma-separated train fractions")
    ls.add_argument("--seeds", default="0,1,2", help="comma-separated seeds (>=3 for 25/50%%)")
    ls.add_argument("--out", default=str(SPLITS_DIR / "lowdata"))
    ls.set_defaults(func=cmd_lowdata_subsets)

    # lowdata-table (aggregate low-data runs, section 9/10)
    lt = sub.add_parser("lowdata-table", help="aggregate low-data runs -> curves (Macro-F1/MAE vs N) + tables")
    lt.set_defaults(func=cmd_lowdata_table)

    return p


def main(argv: list[str] | None = None) -> int:
    # Load .env (HF_TOKEN for gated Llama, etc.) before anything touches the Hub.
    from butppg.orchestrator.env import load_dotenv

    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
