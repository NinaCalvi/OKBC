
import os
import pandas as pd
import pprint
import pickle
import numpy as np
from sklearn import metrics
import argparse
import logging
import itertools
import copy

DATA_PREFIX = ['train', 'val', 'test']
AGG_METRIC = ['mean', 'std', 'min', 'max']


def load_data(base_dir, folds):
    data = {'train': [], 'val': [], 'test': []}
    for i in range(folds):
        for el in DATA_PREFIX:
            ys = pickle.load(
                open(os.path.join(base_dir, el+'_'+str(i)+".pkl"), 'rb'))[:, -1]
            data[el].append(ys)
    return data


def check_data(base_dir, folds):
    for i in range(folds):
        for el in DATA_PREFIX:
            if not os.path.isfile(os.path.join(base_dir, el+'_'+str(i)+".pkl")):
                logging.error("File not present at {}".format(
                    os.path.join(base_dir, el+'_'+str(i)+".pkl")))
                return False
    return True

"""
def make_path(arch, kl, neg_r, rho):
    # exp_c-configs.fb15k_config_7_4.yml_k-0_n--0.5_r-0.01 
    #return "exp_c-{}_k-{}_n-{}_r-{}".format(arch, kl, neg_r, rho)
    return "exp_c-{}_k-{}_n-{}_r-{}".format(arch, kl, neg_r, rho)


def get_params(directory):
    exps = os.listdir(directory)
    neg_rewards = set()
    rhos = set()
    
    kls = set()
    archs = set()
    ex = set()
    for exp in exps:
        if os.path.isdir(os.path.join(directory, exp)):
            delim1 = exp.find("_c-")
            delim2 = exp.find("_k-")
            delim3 = exp.find("_n-")
            delim4 = exp.find("_r-")
            neg_r = exp[delim3+3:delim4]
            rho   = exp[delim4+3:]
            kl    = exp[delim2+3:delim3]
            arch  = exp[delim1+3:delim2]
            neg_rewards.add(neg_r)
            rhos.add(rho)
            kls.add(kl)
            archs.add(arch)
    return (list(sorted(neg_rewards, key=float)), list(sorted(rhos, key=float)), list(sorted(kls, key=float)), list(sorted(archs, key=float)))
"""

def get_string(x):
    return str(x).replace('/','.').replace(' ','.')


def get_params():
    return get_params2()

def get_params2():
    #COPIED FROM CREATE_MULTINODE_JOBS.PY
    neg_reward = [-1, -2]
    rho = [0.1, 0.125]
    config = ['configs/fb15k_config.yml'] 
    kldiv_lambda = [0, 1]
    exclude_t_ids = ['2 5']
    hidden_unit_list = ['90 40','7 5 5 3']
    default_value = [0, -0.05, -0.1]
    #
    names = ['neg_reward','rho','kldiv_lambda','config','exclude_t_ids','hidden_unit_list','default_value']
    all_params = [neg_reward,rho, kldiv_lambda, config,exclude_t_ids,hidden_unit_list,default_value]
    short_names = ['n','r','k','c','ex','hul','df']
    
    assert len(names) == len(all_params)
    assert len(all_params) == len(short_names)
    
    timing_key = 'hidden_unit_list'
    timing = [10]*len(hidden_unit_list)
    #assert(len(globals()[timing_key]) == len(timing))
    assert len(all_params[names.index(timing_key)]) == len(timing),'len of timing should be same as len of timing_key param'
    timing_dict = dict(zip(all_params[names.index(timing_key)],timing))
    all_jobs = list(itertools.product(*all_params))
    
    additional_names = ['train_ml','eval_ml']
    additional_job_list = [
                    [0,0],
                    [1,1]
                    ]
    
    names = names + additional_names
    additional_short_names = ['tml','eml']
    short_names = short_names + additional_short_names
    assert len(names) == len(short_names)
    name2short = dict(zip(names,short_names))
    all_jobs = list(itertools.product(all_jobs,additional_job_list))
    sorted_names = copy.deepcopy(names)
    sorted_names.sort()


    #### WRITTEN AGAIN with MODIFICATION
    for i,key in enumerate(additional_names):
        all_params.append([x[i] for x in additional_job_list])
    #
    name2list = dict(zip(names,all_params))
    for key in sorted_names:
        name2list[key] = [ get_string(x) for x in name2list[key]]
    
    all_settings = {}
    for i, setting in enumerate(all_jobs):
        setting = list(itertools.chain(*setting))
        name_setting = {n: get_string(s) for n, s in zip(names, setting)}
        log_str = '_'.join(['%s-%s' % (name2short[n], name_setting[n]) for n in sorted_names])
        all_settings[log_str] = name_setting 
    #return all_settings, name2list, high level params, rows_cols
    return all_settings, name2list,['config','exclude_t_ids','kldiv_lambda','train_ml','eval_ml','default_value','hidden_unit_list'],['rho','neg_reward']



def get_params1():
    #COPY FROM CREATE_MULTINODE_JOBS.PY
    neg_reward = [-1, -2]
    rho = [0.125]
    config = ['configs/fb15k_config_90_40.yml'] 
    kldiv_lambda = [0, 1]
    exclude_t_ids = ['2 5',None]
    #exclude_t_ids = [None]
    names = ['neg_reward','rho','kldiv_lambda','config','exclude_t_ids']
    all_params = [neg_reward,rho, kldiv_lambda, config,exclude_t_ids]
    short_names = ['n','r','k','c','ex']
    name2list = dict(zip(names,all_params))

    name2short = dict(zip(names,short_names))
    sorted_names = copy.deepcopy(names)
    sorted_names.sort()
    for key in sorted_names:
        name2list[key] = [ get_string(x) for x in name2list[key]]
    
    sorted_names = copy.deepcopy(names)
    all_settings = {}
    sorted_names.sort()
    all_jobs = list(itertools.product(*all_params))
    for i, setting in enumerate(all_jobs):
        setting = list(setting)
        name_setting = {n: get_string(s) for n, s in zip(names, setting)}
        log_str = '_'.join(['%s-%s' % (name2short[n], name_setting[n]) for n in sorted_names])
        all_settings[log_str] = name_setting 
    #return all_settings, name2list, high level params, rows_cols
    return all_settings, name2list,['config','kldiv_lambda','rho'],['exclude_t_ids','neg_reward']


def check_exp(directory, data, folds):
    return check_exp2(directory, data, folds)
    
def check_exp2(directory, data, folds):
    for el in DATA_PREFIX[1:]:
        preds_fname = os.path.join(
                directory, '{}_ml_{}'.format(el,args.eval_ml))
        
        if os.path.isfile(preds_fname):
            return True
        else:
            logging.error("File {} does not exist".format(preds_fname))
            return False
    return True



def check_exp1(directory, data, folds):
    for i in range(folds):
        for el in DATA_PREFIX[1:]:
            if el == 'val':
                preds_fname = os.path.join(
                    directory, 'exp_'+str(i), 'val_pred_ml_{}.txt'.format(args.eval_ml))
            else:
                preds_fname = os.path.join(
                    directory, 'exp_'+str(i), 'test_pred_ml_{}.txt'.format(args.eval_ml))
            if os.path.isfile(preds_fname):
                pred_data = np.loadtxt(preds_fname).tolist()
                if len(data[el][i]) != len(pred_data):
                    logging.error(
                        "Length of {} does not match with ground truth data".format(preds_fname))
                    return False
            else:
                logging.error("File {} does not exist".format(preds_fname))
                return False
    return True


def begin_checks(base_dir, folds, runs):
    data_present = check_data(base_dir, folds)
    if not data_present:
        logging.error("Data Check failed")
        return (False, [])
    data = load_data(base_dir, folds)
    logging.info("Data Check Passed")

    #cols, rows, kls, archs = get_params(os.path.join(base_dir, "run_1"))
    all_settings, name2list,exp_classes, rows_cols = get_params()
    invalid_exps = []
    for run in range(1, runs+1):
        for this_setting in all_settings:
            directory = os.path.join(base_dir, "run_"+str(run), 'exp_'+this_setting)
            if not check_exp(directory, data, folds):
                invalid_exps.append(directory)
    #
    if (len(invalid_exps) == 0):
        logging.info("All experiments are ok")
        return (True, [])
    else:
        logging.error("Following experiments have error\n{}".format(
            str(pprint.pformat(invalid_exps))))
        logging.error("Length is {}".format(len(invalid_exps)))
        return (False, invalid_exps)


def write_invalid(invalid_exps, fname):
    if fname is not None:
        with open(fname, 'w') as f:
            for exp in invalid_exps:
                f.write('/'.join(exp.split('/')[-2:])+'\n')
        logging.error("Written invalid experiments to {}".format(fname))



def calc_exp(directory, data, folds):
    return calc_exp_read(directory, data,folds)


def calc_exp_read(directory, data, folds):
    #directory/test_ml_0
    #directory/test_ml_1
    #directory/val_ml_0
    #directory/val_ml_1
    rv = []
    count = []
    for el in DATA_PREFIX[1:]:
        fname = os.path.join(directory, '{}_ml_{}'.format(el, args.eval_ml))
        f = pd.read_csv(fname,header=None).to_numpy()
        #assert f.shape[0] == 5
        if f.shape[0] != folds:
            print("Num folds: {}. #folds present: {}. ASSUMING 0 performance for them".format(folds, f.shape[0]))
        rv.append(f[:,-1].sum()/folds)
        count.append(f.shape[0])
    #
    return (rv[0], rv[1], count[0], count[1])


def calc_exp_compute(directory, data, folds):
    predictions = {'val': [], 'test': []}
    true = {'val': [], 'test': []}
    logging.info("Calculating results for experiment {}".format(directory))
    for i in range(folds):
        for el in DATA_PREFIX[1:]:
            if el == 'val':
                preds_fname = os.path.join(
                    directory, 'exp_'+str(i), 'valid_preds.txt')
            else:
                preds_fname = os.path.join(
                    directory, 'exp_'+str(i), 'test_preds.txt')
            pred_data = np.loadtxt(preds_fname).tolist()
            predictions[el].extend(pred_data)
            true[el].extend(data[el][i])
    mif_val = metrics.f1_score(true['val'], predictions['val'], labels = args.t_ids, average='micro')
    mif_test = metrics.f1_score(true['test'], predictions['test'], labels= args.t_ids, average='micro')
    return (mif_val, mif_test)



"""
def calc_run_results(directory, rows, cols, KLs, Archs, data, folds):
    results_one_run = {'neg_re': [], 'rho': [], 'val': [], 'test': [], 'kl':[], 'arch':[]}
    for col in cols:
        for row in rows:
        	for kl in KLs:
        		for arch in Archs:
		            exp_directory = os.path.join(directory, make_path(arch, kl, col, row))
		            val, test = calc_exp(exp_directory, data, folds)
		            results_one_run['neg_re'].append(col)
		            results_one_run['rho'].append(row)
		            results_one_run['val'].append(val)
		            results_one_run['test'].append(test)
		            results_one_run['kl'].append(kl)
		            results_one_run['arch'].append(arch)
    df = pd.DataFrame(data=results_one_run)
    return df
"""

def calc_run_results(directory, all_settings, name2list, data, folds):
    results_one_run = None 
    param_keys = None 
    for key in all_settings:
        this_setting = all_settings[key]
        exp_directory = os.path.join(directory,'exp_'+key)
        val, test,val_count, test_count = calc_exp(exp_directory, data, folds)
        if results_one_run is None:
            results_one_run = {}
            param_keys = list(this_setting.keys())
            for this_param in this_setting:
                results_one_run[this_param] = [this_setting[this_param]]
            #
            results_one_run['val'] = [val]
            results_one_run['test'] = [test]
            results_one_run['val_count'] = [val_count]
            results_one_run['test_count'] = [test_count]

        else:
            for this_param in this_setting:
                results_one_run[this_param].append(this_setting[this_param])
            #
            results_one_run['val'].append(val)
            results_one_run['test'].append(test)
            results_one_run['val_count'].append(val_count)
            results_one_run['test_count'].append(test_count)
    df = pd.DataFrame(data=results_one_run)
    return df, param_keys


def write_to_file(table, header, f):
    f.write(header+'\n')
    table.to_csv(f, float_format='%.4f')
    f.write('\n')


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s :: %(asctime)s - %(message)s',
                        level=logging.INFO, datefmt='%d/%m/%Y %I:%M:%S %p')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dir', help="Path of the base directory", required=True)
    parser.add_argument('--folds', type=int, default=5, required=True)
    parser.add_argument('--eval_ml', type=int, default=0, required=True, help='collate results of multi label evaluation?' )
    parser.add_argument('--runs', type=int, default=5, required=True)
    parser.add_argument(
        '--ifile', help='File to write the experiment names which failed', type=str, default=None)

    parser.add_argument('--t_ids', nargs='+', type=int, required=False,default=[1,2,3,4,5,6], help='List of templates')
    args = parser.parse_args()

    ok, invalid_exps = begin_checks(args.dir, args.folds, args.runs)

    if not ok and len(invalid_exps) > 0:
        write_invalid(invalid_exps, args.ifile)
        exit(0)

    logging.info("All Checks passed")
    #cols, rows, KLs, Archs = get_params(os.path.join(args.dir, "run_1"))
    #all_settings, name2list = get_params()
    all_settings, name2list,exp_classes, rows_cols = get_params()

    data = load_data(args.dir, args.folds)
    all_run_results = []
    for run in range(1, args.runs+1):
        logging.info("Calculating results for run {}".format(run))
        df,param_keys = calc_run_results(os.path.join(args.dir, "run_"+str(run)), all_settings, name2list, data, args.folds)
        df['run_id'] = run
        #df = calc_run_results(os.path.join(
        #    args.dir, "run_"+str(run)), rows, cols, KLs, Archs data, args.folds)
        all_run_results.append(df)

    final_result = pd.concat(all_run_results)
    final_result = final_result.reset_index(drop=True) 
    agg_table = final_result.groupby(param_keys).agg({DATA_PREFIX[1]:AGG_METRIC, DATA_PREFIX[2]: AGG_METRIC, DATA_PREFIX[1]+'_count': 'sum', DATA_PREFIX[2]+'_count': 'sum'})
    #agg_table.columns = [d+"_"+a for d,
    #                     a in itertools.product(DATA_PREFIX[1:], AGG_METRIC)]
    
    agg_table.columns = ['_'.join(x) for x in agg_table.columns.to_flat_index()]
    agg_table = agg_table.reset_index()
    agg_table.to_csv(os.path.join(args.dir, 'all_performance.csv'))
    fh = open(os.path.join(args.dir, 'summary.csv'), 'w')

    for this_param in itertools.product(*([name2list[x] for x in exp_classes] + [DATA_PREFIX[1:], AGG_METRIC])):
        d = this_param[-2]
        a = this_param[-1]
        selection = None 
        for ind in range(len(exp_classes)):
            this_selection= agg_table[exp_classes[ind]] == this_param[ind]
            if selection is None:
                selection = this_selection
            else:
                selection = selection & this_selection
        #
        tbl = agg_table[selection].pivot_table(columns = rows_cols[0], index = rows_cols[1], values = d+'_'+a, fill_value = -1)
        print(tbl)
        header = ' '.join(list(map(lambda x: x[0] + '_' + str(x[1]), zip(exp_classes + ['',''], this_param))))

        write_to_file(tbl, header, fh)

    """
    for d, a, kl, arch in itertools.product(DATA_PREFIX[1:], AGG_METRIC, KLs, Archs):
        tbl = agg_table[agg_table["kl"]==kl & agg_table["arch"]==arch].pivot_table(
            columns='neg_re', index='rho', values=d+'_'+a, fill_value=-1)
        print(tbl)
        write_to_file(tbl, (d+'_'+a+'_'+kl+'_'+arch).upper(), fh)
    """

    logging.info("Written results to {}".format(
        os.path.join(args.dir, 'summary.csv')))
