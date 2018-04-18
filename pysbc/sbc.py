import stan_utility
import pystan
import pandas as pd
import numpy as np
from pathos.multiprocessing import Pool, cpu_count

from collections import Counter
import re
import os.path
from functools import wraps, reduce
import time


SEED = 1234
###
# 1. Run DGP model to generate draws. (num_replications,
# 2. Get a fit for each set of data drawn, but can't save them. (control dict,
# 3. Optionally thin
# 4. Rank prior theta within samples
# 5. Return prior thetas, ranks, some fits for

def timed(fn):
    @wraps(fn)
    def timed_wrapped(*args, **kwargs):
        start = time.time()
        r = fn(*args, **kwargs)
        end = time.time()
        print("{0} took {1}s".format(fn.__name__, end-start))
        return r
    return timed_wrapped

def _model_name(filename):
    return os.path.basename(filename).replace(".stan", "")

def _dict_to_filename(dict_):
    if hasattr(dict_, "items"):
        return "(" + "_".join("%s=%s" % (k, _dict_to_filename(v))
                for k, v in dict_.items()) + ")"
    else:
        return dict_

def _params(data, fit_df):
    return set(data.keys()) & (set(fit_df.keys()) - set(["lp__"]))

def _run_result(prior_data, stats, params):
    stats.update((k + "_prior", prior_data[k]) for k in prior_data if k in params)
    return stats

def _compute_param_stat(data, fit_df, stat):
    params = _params(data, fit_df)
    return {k: stat(data[k], fit_df[k]) for k in params}

def _group_params(agg, df):
    params = [re.sub("\[\d+\]$", "", x, 1) for x in df.keys()
            if x.endswith("]")]
    param_counts = Counter(params)
    for param_name, count in param_counts.items():
        df[param_name] = agg([df["{}[{}]".format(param_name, i)]
                for i in range(0, count)])
    return df

def fit_to_df(fit):
    od = fit.extract()
    return od
    #return pd.DataFrame({k: od[k].tolist() for k in od})

def fit_summary(fit):
    summary0 = fit.summary()
    summary = summary0["summary"]
    return pd.DataFrame({"n_eff": [x[-2] for x in summary],
        "rhat": [x[-1] for x in summary]},
        index=summary0["summary_rownames"]).T

####################################
## Handy stat functions for SBC
####################################

def order_stat(prior_val, posterior_samples):
    return np.sum(prior_val < posterior_samples, axis=0)

def mean_stat(prior_val, posterior_samples):
    return np.mean(prior_val < posterior_samples, axis=0)

def rmse_mean(prior, samples):
    return np.sqrt(np.square(np.mean(samples) - prior))

def rmse_averaged(prior, samples):
    return np.sqrt(np.sum(np.square(samples - prior)) / len(samples))



class SBC(object):
    """This class provides SBC's replication statistics facilities. You can pass
    in additional stats to calculate on each replication with stats=[].
    This class can be subclassed to overriding check_fit to look for diagnostics and
    optionally write fits to disk (or wherever). Also can override xform_fit to
    do something like thinning.
    """

    default_stats = [order_stat]

    def __init__(self, dgp_model_name, fit_model_name,
            sampler_args, stats=[], seed=SEED):
        self.fit_model_name = fit_model_name
        self.fit_model = stan_utility.compile_model(fit_model_name)
        self.dgp_model_name = dgp_model_name
        self.dgp_model = stan_utility.compile_model(dgp_model_name)
        self.sampler_args = sampler_args
        self.stats = self.default_stats + stats
        self.seed = seed

    def _map(self, fn, coll):
        print("Using ", cpu_count(), " cores.")
        pool = Pool(cpu_count())
        return pool.imap_unordered(fn, coll, 4)

    #def _map(self, fn, coll): return map(fn, coll)

    def __str__(self):
        return "{cls}_{gen}_{fit}_{args}_seed={seed}".format(
                cls=self.__class__.__name__,
                gen=_model_name(self.dgp_model_name),
                fit=_model_name(self.fit_model_name),
                args=_dict_to_filename(self.sampler_args),
                seed=self.seed)


    @staticmethod
    def _save_fit(fit):
        pass

    @staticmethod
    def _gen_data_iter(original_data, datasets):
        params = list(datasets.keys())
        for i in range(len(datasets[params[0]])):
            og = original_data.copy()
            og.update({k: datasets[k][i] for k in params})
            yield og

    def run_DGP(self, data, num_datasets):
        fit = self.dgp_model.sampling(data=data, iter=num_datasets,
                warmup=0, chains=1, algorithm='Fixed_param',
                seed=self.seed)
        return self._gen_data_iter(data, fit.extract())

    def fit_data(self, fit_data, sampler_args=None):
        if sampler_args:
            kwargs = self.sampler_args.copy()
            kwargs.update(sampler_args)
        else:
            kwargs = self.sampler_args
        return self.fit_model.sampling(data=fit_data, seed=self.seed, **kwargs)

    def check_fit(self, fit, summary):
        #XXX use stan_utility diagnostics here to check.
        pass

    def xform_fit(self, fit, summary): # could be used for thinning
        return fit, fit_to_df(fit), summary

    def compute_stats(self, data, fit_df):
        stat_dicts = [{"{}_{}".format(k, stat.__name__): v
                for k, v in _compute_param_stat(data, fit_df, stat).items()}
            for stat in self.stats]
        return reduce(lambda a, e: a.update(e) or a, stat_dicts, {})

    def get_summary_stats(self, summary, pars):
        result = {}
        for p in pars:
            flatnames = [p2 for p2 in summary if p2.startswith(p + "[")]
            result[p + "_rhat"] = [summary[fn]["rhat"] for fn in flatnames]\
                    or summary[p][1]
            result[p + "_n_eff"] = [summary[fn]['n_eff'] for fn in flatnames]\
                    or summary[p][0]
        return result

    #@timed
    def replication(self, fit_data):
        fit = self.fit_data(fit_data)
        summary = fit_summary(fit)
        self.check_fit(fit, summary)
        fit, df, summary = self.xform_fit(fit, summary)
        stats = self.compute_stats(fit_data, df)
        params = _params(fit_data, df)
        summary_stats = self.get_summary_stats(summary, params)
        stats.update(summary_stats)
        return _run_result(fit_data, stats, params)

    @timed
    def run(self, data, num_replications):
        gdata = self.run_DGP(data, num_replications)
        return pd.DataFrame(list(self._map(self.replication, gdata)))

class CGR(SBC):
    default_stats = [mean_stat]


class ThinSBC(SBC):
    def __init__(self, desired_ranks, *args, **kwargs):
        self.desired_ranks = desired_ranks
        super().__init__(*args, **kwargs)

    def xform_fit(self, fit, summary):
        N = fit.sim["iter"]
        skip = N / summary.loc["n_eff"]
        _group_params(np.max, skip) # XXX Choice of aggregation here
        needed = int(max(skip * self.desired_ranks + fit.sim["warmup"]))
        if needed > N:
            print("Redoing! needed {}".format(needed))
            self.sampler_args
            fit = self.fit_data(fit.data, sampler_args=dict(iter=needed))
            summary = fit_summary(fit)
            df = fit_to_df(fit)
            for p in df.keys():
                df[p] = df[p][np.arange(0, len(df[p]), int(skip[p]))]
        else:
            df = fit_to_df(fit)
        for p in df.keys():
            df[p] = df[p][:self.desired_ranks]
        return fit, df, summary


def vanilla_sbc_8schools(num_reps):
    school_data = dict(J=8, K=2, sigma = [15, 10, 16, 11,  9, 11, 10, 18])
    sbc = SBC("../code/gen_8schools.stan", "../code/8schools.stan",
            dict(chains=1, iter=2000, control=dict(adapt_delta=0.98)),
            stats=[rmse_mean, rmse_averaged])
    stats = sbc.run(school_data, num_reps)
    timed(stats.to_csv)(str(sbc) + ".csv")
    print(stats.head())

def thin_sbc_8schools(num_reps):
    school_data = dict(J=8, K=2, sigma = [15, 10, 16, 11,  9, 11, 10, 18])
    sbc = ThinSBC(1000,
            "../code/gen_8schools.stan", "../code/8schools.stan",
            dict(chains=1, iter=9000, warmup=1000,
                control=dict(adapt_delta=0.98)),
            stats=[rmse_mean, rmse_averaged])
    stats = sbc.run(school_data, num_reps)
    timed(stats.to_csv)(str(sbc) + ".csv")
    print(stats.head())

def thin_lin_regr_wide(num_reps):
    N=25
    data = dict(N=N, X=np.random.normal(0, 5, N))
    sbc = ThinSBC(1000, "../code/gen_lin_regr_c.stan", "../code/lin_regr_c_wide.stan",
            dict(chains=1, iter=5000, warmup=1000), stats=[rmse_mean, rmse_averaged])
    reps = sbc.run(data, num_reps)
    timed(reps.to_csv)(str(sbc) + ".csv")


if __name__ == "__main__":
    #### SBC 8
    import sys
    num_reps = int(sys.argv[1])
    thin_lin_regr_wide(num_reps)
