"""Rebuild corrected bootstrap replications before notebook 03 consumes them.

From the project root:
    PYTHONPATH=src python -m cdr.bootstrap_rebuild --publish

Without --publish, outputs are candidates under scratch/ and existing numerical
results are not replaced. --accept-reference-update additionally updates only
the three regression-reference values affected by the baseline-refit correction.
It never adjusts a result to recover significance or to match an old interval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from . import dormant as dm, memorydecay as md, paths
from .util import sha256

_MC_NAME = 'dormant_pseudoevent_monte_carlo_replications.csv'
_BOOT_NAME = 'dormant_integrated_bootstrap_replications.csv'
_worker_inputs = None


def _worker_init(panel, wide, first_year_map, fits):
    global _worker_inputs
    _worker_inputs = (panel, wide, first_year_map, fits)


def _worker(task):
    start, size = task
    return dm.integrated_bootstrap(*_worker_inputs, n_rep=size, start_rep=start)


def _save_csv(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.tmp')
    frame.to_csv(temp, index=False)
    os.replace(temp, target)


def _save_json(obj, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.tmp')
    def clean(value):
        if isinstance(value, dict): return {k:clean(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)): return [clean(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
        if isinstance(value, np.generic): return value.item()
        return value
    temp.write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, target)


def load_inputs():
    panel_path = paths.OUT_DATA / 'spotify_song_snapshot_panel.csv'
    master_path = paths.OUT_DATA / 'master_song_panel.csv'
    for p in (panel_path, master_path):
        if not p.is_file():
            raise FileNotFoundError(f'Missing {p}. Run notebooks 00 and 01 first, or FULL_INFERENCE=1 ./run_all.sh.')
    panel = pd.read_csv(panel_path, low_memory=False)
    master = pd.read_csv(master_path, low_memory=False)
    # Exact replication of notebook 03, not a fit to rounded parameter exports.
    panel = panel.loc[panel['age'] >= 0].copy()
    fits = {}
    for snap in dm.SNAP_NAMES:
        agg = md.aggregate_age_means(panel.loc[panel.film_linked.eq(0) & panel.snapshot.eq(snap)])
        fits[snap] = md.fit_canonical_gridsearch(agg)
    panel = md.apply_expected_excess(panel, fits)
    panel['song_artist_id'] = panel['song'].astype(str) + '||' + panel['artist'].astype(str)
    linked = master.loc[master.film_linked.eq(1)].dropna(subset=['first_year']).copy()
    linked['record_id'] = linked['song'].astype(str) + '||' + linked['artist'].astype(str)
    if linked.record_id.duplicated().any():
        raise ValueError('Duplicate exact Billboard records in the first-film-year map.')
    first = linked.set_index('record_id')['first_year']
    if panel.duplicated(['song_artist_id','snapshot']).any():
        raise ValueError('Duplicate exact Billboard song-snapshot records.')
    wide = dm.build_snapshot_wide(panel)
    source_hashes = {p.name: sha256(p) for p in (panel_path, master_path)}
    return panel, master, wide, first, fits, source_hashes


def identity_refit_check(panel, fits) -> pd.DataFrame:
    ids = panel.loc[panel.film_linked.eq(0), 'song_artist_id'].drop_duplicates().to_numpy()
    sample = dm._resampled_never_film_panel(panel, ids)
    for snap in dm.SNAP_NAMES:
        a = md.aggregate_age_means(panel.loc[panel.film_linked.eq(0) & panel.snapshot.eq(snap)])
        b = md.aggregate_age_means(sample.loc[sample.snapshot.eq(snap)])
        assert_frame_equal(a, b, check_exact=True)
    refits = dm.refit_memorydecay_bootstrap(panel, None, fits, sampled_ids=ids)
    rows=[]
    for snap in dm.SNAP_NAMES:
        age = panel.loc[panel.snapshot.eq(snap),'age'].to_numpy(float)
        original = md.canonical_curve(age, **{k:fits[snap][k] for k in ('N','p','r','q')})
        refit = md.canonical_curve(age, **{k:refits[snap][k] for k in ('N','p','r','q')})
        np.testing.assert_allclose(original,refit,rtol=1e-12,atol=1e-10)
        if fits[snap]['n_starts'] != refits[snap]['n_starts']:
            raise AssertionError('Bootstrap and primary starting grids differ.')
        rows.append(dict(snapshot=snap,max_prediction_difference=float(np.max(np.abs(original-refit))),
                         primary_n_starts=fits[snap]['n_starts'],refit_n_starts=refits[snap]['n_starts']))
    return pd.DataFrame(rows)


def _check_existing_point_estimates(mc, panel, master):
    """The refit correction must not silently alter the point-estimate design."""
    reference = paths.DATA / 'expected_results.csv'
    if not reference.exists():
        raise FileNotFoundError('Publication requires data/expected_results.csv.')
    ref = pd.read_csv(reference).set_index('result_id')
    values = {
        'billboard_universe':len(master),
        'film_linked_records':int(master.film_linked.eq(1).sum()),
        'expected_memory_observations':len(panel),
        'dormant_excess':float(mc.excess_attention_estimate.mean()),
        'dormant_exit_pp':float(mc.exit_difference_pp.mean()),
        'pseudo_event_positive_assignments':int((mc.excess_attention_estimate>0).sum()),
    }
    for key,value in values.items():
        if key not in ref.index:
            raise ValueError(f'Missing unchanged point-estimate reference: {key}.')
        r=ref.loc[key]
        if abs(value-float(r.expected_value))>float(r.tolerance):
            raise RuntimeError(f'Unchanged-design check failed for {key}: {value} versus {r.expected_value}. '
                               'Do not attribute this difference only to the bootstrap refitter.')
    return ref


def run(args) -> Path:
    if args.accept_reference_update and not args.publish:
        raise ValueError('--accept-reference-update requires --publish.')
    if args.publish and (args.assignments!=500 or args.replications!=250):
        raise ValueError('Publishing requires the existing contract: 500 assignments and 250 bootstrap replications.')
    t0=time.perf_counter()
    panel,master,wide,first,fits,source_hashes=load_inputs()
    config=dict(bootstrap_refit_version=dm.BOOTSTRAP_REFIT_VERSION,
                code_sha256=dm._inference_code_hashes(),runner_sha256=sha256(Path(__file__)),
                input_sha256=source_hashes,
                assignments=args.assignments,replications=args.replications,batch_size=args.batch_size,
                support_fingerprint=md.dormant_support_fingerprint(panel))
    digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    out=paths.PROJECT_ROOT/'scratch'/'bootstrap_refit_2026_09_10'/digest
    out.mkdir(parents=True,exist_ok=True)
    config_path=out/'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise RuntimeError('Checkpoint configuration does not match current inputs and code.')
    _save_json(config,config_path)
    # Recheck actual primary fits even when the heavy bootstrap is resumed.
    checks=identity_refit_check(panel,fits)
    _save_csv(checks,out/'identity_refit_check.csv')
    print(checks.to_string(index=False),flush=True)
    mc=dm.monte_carlo_pseudoevent(wide,first,n_rep=args.assignments)
    _save_csv(mc,out/_MC_NAME)
    if args.publish:
        _check_existing_point_estimates(mc,panel,master)
    tasks=[];parts=[]
    for start in range(1,args.replications+1,args.batch_size):
        size=min(args.batch_size,args.replications-start+1)
        p=out/f'bootstrap_{start:04d}_{start+size-1:04d}.csv'
        if p.exists():
            stored=pd.read_csv(p, float_precision="round_trip")
            if (len(stored)!=size or 'bootstrap_refit_version' not in stored
                    or not stored.bootstrap_refit_version.eq(dm.BOOTSTRAP_REFIT_VERSION).all()
                    or stored.replication.tolist()!=list(range(start,start+size))
                    or stored.seed.tolist()!=list(range(dm.BOOT_SEED_START+start-1,dm.BOOT_SEED_START+start-1+size))):
                raise RuntimeError(f'Invalid checkpoint {p}. Remove it only after inspecting the failure.')
            parts.append(stored)
        else:
            tasks.append((start,size))
    def store(frame):
        start=int(frame.replication.min());end=int(frame.replication.max())
        _save_csv(frame,out/f'bootstrap_{start:04d}_{end:04d}.csv')
        parts.append(frame)
        print(f'Completed {sum(len(f) for f in parts)}/{args.replications} corrected replications',flush=True)
    if args.jobs==1:
        for start,size in tasks:
            store(dm.integrated_bootstrap(panel,wide,first,fits,n_rep=size,start_rep=start))
    elif tasks:
        with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_worker_init,initargs=(panel,wide,first,fits)) as pool:
            futures=[pool.submit(_worker,task) for task in tasks]
            for future in as_completed(futures):store(future.result())
    boot=pd.concat(parts,ignore_index=True).sort_values('replication').reset_index(drop=True)
    dm._check_replication_rows(mc,boot)
    _save_csv(boot,out/_BOOT_NAME)
    summary=dm.summarize_stored_inference(mc,boot)
    values=pd.DataFrame(list(summary.items()),columns=['quantity','value'])
    _save_csv(values,out/'dormant_integrated_inference.csv')
    old_summary=paths.RESULTS/'dormant_integrated_inference.csv'
    if old_summary.exists():
        change=pd.read_csv(old_summary).rename(columns={'value':'previous_value'}).merge(
            values.rename(columns={'value':'corrected_value'}),on='quantity',how='outer')
        change['difference']=change.corrected_value-change.previous_value
        _save_csv(change,out/'numerical_changes.csv')
    _save_json(dict(status='completed',scope='baseline-refitter consistency only',
                    input_scope='project-local files listed in run_config.json',published=False,elapsed_seconds=time.perf_counter()-t0,
                    summary=summary,manuscript_synchronized=False),out/'result.json')
    if args.publish:
        if args.accept_reference_update:
            ref_path=paths.DATA/'expected_results.csv';ref=pd.read_csv(ref_path)
            updates={'dormant_excess_ci_low':summary['dormant_excess_integrated_ci_low'],
                     'dormant_excess_ci_high':summary['dormant_excess_integrated_ci_high'],
                     'decay_equivalent_years':summary['dormant_decay_equivalent_years']}
            if not np.isfinite(list(updates.values())).all():
                raise RuntimeError('A decay translation is not defined. Review it rather than replacing its numerical reference with NaN.')
            for key,value in updates.items():
                if ref.result_id.eq(key).sum()!=1:raise ValueError(f'Invalid reference row: {key}')
                ref.loc[ref.result_id.eq(key),'expected_value']=value
        # Keep every overwritten file. Old percentile endpoints are not reused.
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup=out/('before_publish_'+stamp);backup.mkdir()
        for name in (_MC_NAME,_BOOT_NAME,'inference_metadata.json'):
            p=paths.INFERENCE/name
            if p.exists():shutil.copy2(p,backup/name)
        dm.write_inference_files(mc,boot,config['support_fingerprint'])
        dm.validate_inference_files(config['support_fingerprint'])
        if args.accept_reference_update:
            shutil.copy2(ref_path,backup/'expected_results.csv')
            _save_csv(ref,ref_path)
        report=json.loads((out/'result.json').read_text());report.update(published=True,backup=str(backup),
            numerical_reference_updated=args.accept_reference_update)
        _save_json(report,out/'result.json')
        print('Corrected inference files published with a rollback copy. Figures/tables must be regenerated by notebook 03.',flush=True)
        if not args.accept_reference_update:
            print('Existing numerical reference endpoints were preserved. Notebook 06 may stop until the changed endpoints are reviewed.',flush=True)
    print(values.to_string(index=False),flush=True)
    print(f'Results: {out}\nManuscript text and Overleaf were not modified.',flush=True)
    return out


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--assignments',type=int,default=500)
    p.add_argument('--replications',type=int,default=250)
    p.add_argument('--jobs',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=10)
    p.add_argument('--publish',action='store_true')
    p.add_argument('--accept-reference-update',action='store_true')
    return p


def main():
    p=parser();args=p.parse_args()
    if min(args.assignments,args.replications,args.jobs,args.batch_size)<1:p.error('Counts must be positive.')
    run(args)

if __name__=='__main__':main()
