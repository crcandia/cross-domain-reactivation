"""Synthetic regression tests. Run: PYTHONPATH=src python -m cdr.bootstrap_refit_tests

The production optimizer is exercised in two identity-refit tests. Faster tests
mock only the optimizer to isolate sampling, failure handling, and provenance.
No test asserts empirical significance or nominal interval coverage.
"""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from . import dormant as dm, memorydecay as md, paths
from .util import SNAPSHOTS


def mocked_fit(agg):
    return dict(N=85.,p=.27434,r=.53539,q=.02489,n_starts=195,sse_log_weighted=0.)


def synthetic_panel():
    rows=[]
    for j in range(96):
        treated=int(j>=72); cyear=1958+j%50
        for k,(snap,year,_,_) in enumerate(SNAPSHOTS):
            age=year-cyear
            expected=float(md.canonical_curve([age],85.,.27434,.53539,.02489)[0])
            attention=max(.1,expected+(j%9-4)*.8+.1*k+treated*(k>=2)*4)
            rows.append(dict(song_artist_id=f'Song {j}||Artist {j%15}',song=f'Song {j}',artist=f'Artist {j%15}',
                snapshot=snap,snapshot_year=year,film_linked=treated,age=age,attention=attention,
                expected_attention=expected,excess_attention=attention-expected,chart_year=cyear,
                weeks=10+j%12,peak_pos=1+j%40,artist_catalog_size_excl_song=2+j%8,
                artist_total_weeks_excl_song=10+j%25,artist_superstar_top1pct=int(j%17==0)))
    return (pd.DataFrame(rows),{s:mocked_fit(None) for s,*_ in SNAPSHOTS},
            pd.Series({f'Song {j}||Artist {j%15}':2018. for j in range(72,96)}))


class RefitTests(unittest.TestCase):
    def setUp(self):
        self.panel,self.fits,self.first=synthetic_panel()
        self.ids=self.panel.loc[self.panel.film_linked.eq(0),'song_artist_id'].drop_duplicates().to_numpy()

    def test_identity_inputs(self):
        sample=dm._resampled_never_film_panel(self.panel,self.ids)
        for snap in dm.SNAP_NAMES:
            a=md.aggregate_age_means(self.panel.query('film_linked == 0 and snapshot == @snap'))
            b=md.aggregate_age_means(sample[sample.snapshot.eq(snap)])
            assert_frame_equal(a,b,check_exact=True)

    def test_whole_trajectory_multiplicity(self):
        sample=dm._resampled_never_film_panel(self.panel,[self.ids[0],self.ids[0],self.ids[1],self.ids[3]])
        counts=sample.groupby(['song_artist_id','snapshot']).size()
        self.assertTrue((counts.loc[self.ids[0]]==2).all()); self.assertEqual(len(sample),16)

    def test_floor_and_fractional_age(self):
        rows=pd.DataFrame(dict(song_artist_id=['a','b','c','d'],snapshot=['2016-10']*4,film_linked=[0]*4,
                               age=[.8,.9,1.8,2.8],attention=[10.,20.,30.,40.]))
        a=md.aggregate_age_means(dm._resampled_never_film_panel(rows,['a','a','b','c','d']))
        self.assertEqual(list(a.age_int),[0,1,2]);self.assertAlmostEqual(a.age.iloc[0],(.8*2+.9)/3)
        self.assertAlmostEqual(a.attention.iloc[0],40/3)

    def test_zero_mean_cells_same_rule(self):
        p=self.panel.copy();p.loc[p.age<10,'attention']=0.
        sample=dm._resampled_never_film_panel(p,self.ids)
        for snap in dm.SNAP_NAMES:
            assert_frame_equal(md.aggregate_age_means(p.query('film_linked == 0 and snapshot == @snap')),
                               md.aggregate_age_means(sample[sample.snapshot.eq(snap)]),check_exact=True)

    def test_rng_consumption(self):
        a=np.random.default_rng(123);b=np.random.default_rng(123)
        with patch.object(dm,'fit_canonical_gridsearch',side_effect=mocked_fit):
            dm.refit_memorydecay_bootstrap(self.panel,a,self.fits)
        b.choice(self.ids,size=len(self.ids),replace=True)
        self.assertEqual(a.integers(0,2**31),b.integers(0,2**31))

    def test_primary_called_not_legacy(self):
        with patch.object(md,'fit_canonical_bounded',side_effect=AssertionError('legacy')):
            with patch.object(dm,'fit_canonical_gridsearch',side_effect=mocked_fit) as fitter:
                result=dm.refit_memorydecay_bootstrap(self.panel,None,{},sampled_ids=self.ids)
                self.assertEqual(fitter.call_count,4);self.assertGreater(result['2016-10']['r'],.3)
                for call in fitter.call_args_list:self.assertIn('age_int',call.args[0])

    def test_real_full_grid_identity(self):
        snap='2016-10';a=md.aggregate_age_means(self.panel.query('film_linked == 0 and snapshot == @snap'))
        expected=md.fit_canonical_gridsearch(a)
        with patch.object(dm,'SNAP_NAMES',[snap]):
            actual=dm.refit_memorydecay_bootstrap(self.panel,None,self.fits,sampled_ids=self.ids)[snap]
        self.assertEqual(actual,expected)
        self.assertEqual(actual['n_starts'],len(md._initial_N_values(a.attention))*len(md.PRQ_STARTS))
        t=np.linspace(0,65,200)
        np.testing.assert_array_equal(md.canonical_curve(t,**{k:actual[k] for k in ['N','p','r','q']}),
                                      md.canonical_curve(t,**{k:expected[k] for k in ['N','p','r','q']}))

    def test_real_near_coincident_rates(self):
        snap='2022-08';p=self.panel.query('film_linked == 0 and snapshot == @snap').copy()
        p['attention']=md.canonical_curve(p.age.to_numpy(),66.38467,.01362,.01853,.03212)
        expected=md.fit_canonical_gridsearch(md.aggregate_age_means(p))
        with patch.object(dm,'SNAP_NAMES',[snap]):
            actual=dm.refit_memorydecay_bootstrap(p,None,{},sampled_ids=self.ids)[snap]
        self.assertEqual(actual,expected);self.assertLess(actual['p'],.3)

    def test_duplicate_snapshot_rejected(self):
        with self.assertRaisesRegex(ValueError,'one row'):
            dm._resampled_never_film_panel(pd.concat([self.panel,self.panel.iloc[:1]]),self.ids)

    def test_invalid_ids_rejected(self):
        for sample in ([],[None],['unknown']):
            with self.assertRaises(ValueError):dm._resampled_never_film_panel(self.panel,sample)

    def test_nonfinite_and_negative_rejected(self):
        for val in [np.nan,np.inf,-1.]:
            p=self.panel.copy();p.loc[0,'age']=val
            with self.assertRaisesRegex(ValueError,'finite and nonnegative'):dm._resampled_never_film_panel(p,self.ids)

    def test_failed_fit_does_not_fallback(self):
        with patch.object(dm,'fit_canonical_gridsearch',side_effect=ValueError('fail')):
            with self.assertRaisesRegex(RuntimeError,'Primary-estimator refit failed'):
                dm.refit_memorydecay_bootstrap(self.panel,None,{},sampled_ids=self.ids)

    def test_missing_snapshot_rejected(self):
        with patch.object(dm,'fit_canonical_gridsearch',side_effect=mocked_fit):
            with self.assertRaisesRegex(RuntimeError,'No observations'):
                dm.refit_memorydecay_bootstrap(self.panel[self.panel.snapshot.ne('2025')],None,{},sampled_ids=self.ids)

    def test_batch_seeds_results(self):
        w=dm.build_snapshot_wide(self.panel)
        with patch.object(dm,'fit_canonical_gridsearch',side_effect=mocked_fit):
            a=dm.integrated_bootstrap(self.panel,w,self.first,self.fits,n_rep=4)
            b=pd.concat([dm.integrated_bootstrap(self.panel,w,self.first,self.fits,n_rep=2,start_rep=k)
                         for k in [1,3]],ignore_index=True)
        assert_frame_equal(a,b,check_exact=True)

    def test_failed_replication_reports_seed(self):
        w=dm.build_snapshot_wide(self.panel)
        with patch.object(dm,'fit_canonical_gridsearch',side_effect=ValueError('fail')):
            with self.assertRaisesRegex(RuntimeError,f'replication 3, seed {dm.BOOT_SEED_START+2}'):
                dm.integrated_bootstrap(self.panel,w,self.first,self.fits,n_rep=1,start_rep=3)

    def test_input_unchanged(self):
        before=self.panel.copy(deep=True)
        with patch.object(dm,'fit_canonical_gridsearch',side_effect=mocked_fit):
            dm.refit_memorydecay_bootstrap(self.panel,np.random.default_rng(3),self.fits)
        assert_frame_equal(before,self.panel,check_exact=True)

    def test_cache_roundtrip_stale_rejected(self):
        w=dm.build_snapshot_wide(self.panel);mc=dm.monte_carlo_pseudoevent(w,self.first,n_rep=3)
        with patch.object(dm,'fit_canonical_gridsearch',side_effect=mocked_fit):
            boot=dm.integrated_bootstrap(self.panel,w,self.first,self.fits,n_rep=3)
        with tempfile.TemporaryDirectory() as temp,patch.object(paths,'INFERENCE',Path(temp)):
            meta=dm.write_inference_files(mc,boot,'synthetic-support')
            self.assertEqual(dm.validate_inference_files('synthetic-support'),meta)
            with self.assertRaisesRegex(RuntimeError,'fingerprint mismatch'):dm.validate_inference_files('wrong')
            meta['bootstrap_refit_version']='legacy';(Path(temp)/'inference_metadata.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(RuntimeError,'obsolete bootstrap'):dm.validate_inference_files('synthetic-support')

    def test_legacy_mixed_failed_and_duplicate_rows_rejected(self):
        w=dm.build_snapshot_wide(self.panel);mc=dm.monte_carlo_pseudoevent(w,self.first,n_rep=3)
        with patch.object(dm,'fit_canonical_gridsearch',side_effect=mocked_fit):
            b=dm.integrated_bootstrap(self.panel,w,self.first,self.fits,n_rep=3)
        for bad in [b.drop(columns='bootstrap_refit_version'),
                    b.assign(bootstrap_refit_version=['legacy']+[dm.BOOTSTRAP_REFIT_VERSION]*2),
                    b.assign(dormant_excess_attention=[np.nan,1.,2.]),b.assign(replication=[1,1,3])]:
            with self.assertRaises(RuntimeError):dm.summarize_stored_inference(mc,bad)


    def test_table_legacy_cache_rejected(self):
        from . import tex
        with tempfile.TemporaryDirectory() as temp, patch.object(paths, 'INFERENCE', Path(temp)):
            with self.assertRaisesRegex(RuntimeError, 'Regenerate corrected inference'):
                tex._current_bootstrap_table_summary()
            (Path(temp)/'inference_metadata.json').write_text(json.dumps({
                'dormant_analytical_support_fingerprint':'synthetic', 'files':[]}))
            with self.assertRaisesRegex(RuntimeError, 'obsolete bootstrap'):
                tex._current_bootstrap_table_summary()

    def test_table_values_and_native_note(self):
        from . import tex
        w=dm.build_snapshot_wide(self.panel)
        mc=dm.monte_carlo_pseudoevent(w,self.first,n_rep=3)
        with patch.object(dm,'fit_canonical_gridsearch',side_effect=mocked_fit):
            boot=dm.integrated_bootstrap(self.panel,w,self.first,self.fits,n_rep=3)
        summary=dm.summarize_stored_inference(mc,boot)
        labels={
            r'Integrated bootstrap 95\% CI, low':'dormant_excess_integrated_ci_low',
            r'Integrated bootstrap 95\% CI, high':'dormant_excess_integrated_ci_high',
            r'Above-threshold integrated 95\% CI, low (pp)':'exit_integrated_ci_low',
            r'Above-threshold integrated 95\% CI, high (pp)':'exit_integrated_ci_high',
            'Decay-equivalent years (median)':'dormant_decay_equivalent_years',
            r'Decay-equivalent 95\% CI, low':'dormant_decay_integrated_ci_low',
            r'Decay-equivalent 95\% CI, high':'dormant_decay_integrated_ci_high',
        }
        rows=[[label,tex.fmt_num(summary[key],3)] for label,key in labels.items()]
        with patch.object(tex,'_current_bootstrap_table_summary',return_value=summary):
            output=tex._publication_fields('test','tab:si-dormant-integrated-inference',
                                         ['Quantity','Value'],rows,'Existing note.')
        self.assertIn('same floor-age aggregation',output[-1]);self.assertNotIn('bounded',output[-1])
        stale=[r.copy() for r in rows];stale[0][1]='999'
        with self.assertRaisesRegex(RuntimeError,'Stale bootstrap table'):
            tex._check_bootstrap_table_values(stale,summary)
        with self.assertRaisesRegex(RuntimeError,'Incomplete'):
            tex._check_bootstrap_table_values(rows[:-1],summary)

if __name__=='__main__':unittest.main(verbosity=2)
