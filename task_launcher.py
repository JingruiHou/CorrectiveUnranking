import task_utils
from unranking_task import UnRankingTask
import torch


def run_retrain_exp(from_initial=True):
    retraining_epoch = dict(trec=15, marco=15)
    for data_name in data_names:
        for model_cfg_path in model_cfg_paths:
            config = dict(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            model_config = task_utils.loadyaml(f'{model_cfg_path}')
            data_config = task_utils.loadyaml(f'./config/data.{data_name}.yaml')
            config.update(data_config)
            config.update(model_config)
            config['forgetting_ratio'] = 0.1
            config['num_workers'] = 0
            if from_initial:
                config['task_prefix'] = 'retrain'
                config['trained_model_epoch'] = 0
            else:
                config['task_prefix'] = 'catastrophic'
                config['trained_model_epoch'] = trained_model_epoch[data_name][config['model_name']]
            config['unlearn_epoch'] = retraining_epoch[data_name]
            task = UnRankingTask(config)
            task.do_retrain()
            for epoch in range(1, config['unlearn_epoch'] + 1, 2):
                task.do_test(test='trained', epoch=epoch)
                task.do_test(test='dev', epoch=epoch)


def run_correcting_exp():
    num_neg_docs_used = 5
    unlearn_batch_size = 64
    quantile_neg = 0.0
    hinge_loss_cfg = {
        'hinge_loss_retain_pos_a': -1,
        'hinge_loss_retain_pos_b': -2,
        'hinge_loss_retain_neg_a': 1,
        'hinge_loss_retain_neg_b': 1,
        'hinge_loss_forget_pos_a': 1,
        'hinge_loss_forget_pos_b': 1,
        'hinge_loss_forget_neg_a': -1,
        'hinge_loss_forget_neg_b': -5,
    }
    retraining_epoch = dict(trec=15, marco=15)
    filters = ['bert_model.embeddings', 'bert_model.transformer']
    for data_name in data_names:
        for model_cfg_path in model_cfg_paths:
            config = dict(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            model_config = task_utils.loadyaml(f'{model_cfg_path}')
            data_config = task_utils.loadyaml(f'./config/data.{data_name}.yaml')
            config['parameter_name_filter'] = filters
            config.update(data_config)
            config.update(model_config)
            config['batch_size'] = unlearn_batch_size
            config['forgetting_ratio'] = 0.1
            config['num_workers'] = 0
            config['task_prefix'] = 'correction_v4'
            config['teacher_model'] = True
            config['num_neg_docs_used'] = num_neg_docs_used
            config['trained_model_epoch'] = trained_model_epoch[data_name][config['model_name']]
            config['learning_rate'] = 0.00005
            config['quantile_neg'] = quantile_neg
            config.update(hinge_loss_cfg)
            config['unlearn_epoch'] = retraining_epoch[data_name]
            task = UnRankingTask(config)
            task.do_retrain_correction()
            for epoch in range(1, config['unlearn_epoch'] + 1, 2):
                task.do_test(test='trained', epoch=epoch)
                task.do_test(test='dev', epoch=epoch)


if __name__ == '__main__':
    torch.cuda.set_device(7)
    data_names = ['marco', 'trec']
    model_cfg_paths = ['config/ranker.bertcat.yaml', 'config/ranker.colbert.yaml',
                       'config/ranker.bertdot.yaml', 'config/ranker.parade.yaml'][:2]
    trained_model_epoch = {
        'trec': dict(BERTCat=8, BERTdot=8, ColBERT=4, Parade=8),
        'marco': dict(BERTCat=3, BERTdot=3, ColBERT=3, Parade=3)
    }
    run_retrain_exp(from_initial=False)
    # run_correcting_exp()
