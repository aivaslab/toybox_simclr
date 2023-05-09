import argparse
import numpy as np

import transfer_learner

TR_LRNR_HT = 0
TR_LRNR_EV = 1


def get_parser(desc):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', '-m', required=True, type=str)
    parser.add_argument('--lrs', '-lrs', required=True, nargs='+', type=float)
    parser.add_argument('--epochs', '-e', required=True, type=int)
    parser.add_argument('--fraction', '-f', default=1.0, type=float)
    parser.add_argument('--dataset', '-data', default="cifar10", type=str)
    parser.add_argument('--combined', '-c', default=False, action="store_true")
    parser.add_argument('--batch-size', '-b', default=128, type=int)
    parser.add_argument('--num-reps', '-n', default=3, type=int)
    parser.add_argument('--layers', '-ls', nargs='+', type=int)
    parser.add_argument('--type', '-t', default=TR_LRNR_HT, type=int)
    parser.add_argument('--decay_epochs', '-de', default=20, type=int)
    
    return parser.parse_args()


def run_transfer_ht(exp_args):
    assert len(exp_args['layers']) == 1
    exp_args['num_layers_frozen'] = exp_args['layers'][0]
    results_dict = {}
    for lr in exp_args["lrs"]:
        exp_args['lr'] = lr
        exp_args['hypertune'] = True
        print("=================================================================================================")
        print("learning rate: ", exp_args['lr'])
        train_accs, test_accs = transfer_learner.run_transfer_learner_reps(exp_args=exp_args)
        results_dict[lr] = (train_accs, test_accs)
        print("=================================================================================================")
    print("lr".ljust(7), "train_mean".ljust(12), "train_std".ljust(12), "test_mean".ljust(12), "test_std".ljust(12))
    max_test_acc = -0.1
    max_lr = -0.1
    for lr in exp_args["lrs"]:
        train_mean = np.mean(np.asarray(results_dict[lr][0]))
        train_sd = np.std(np.asarray(results_dict[lr][0]))
        test_mean = np.mean(np.asarray(results_dict[lr][1]))
        test_sd = np.std(np.asarray(results_dict[lr][1]))
        print('{0:<7.4f} {1:<12.2f} {2:<12.2f} {3:<12.2f} {4:<12.2f}'.format(lr, train_mean, train_sd, test_mean,
                                                                             test_sd))
        if test_mean > max_test_acc:
            max_lr = lr
            max_test_acc = test_mean
    print("We got max test acc. of {0:.2f} with lr {1:.5f}".format(max_test_acc, max_lr))
    print("=================================================================================================")
    exp_args['lr'] = max_lr
    exp_args['hypertune'] = False
    transfer_learner.run_transfer_learner_reps(exp_args=exp_args)


def run_transfer_ev(exp_args):
    assert len(exp_args['lrs']) == 1
    exp_args['lr'] = exp_args['lrs'][0]
    exp_args['hypertune'] = False
    results_dict = {}
    for l in exp_args['layers']:
        exp_args['num_layers_frozen'] = l
        print("=================================================================================================")
        print("Layers frozen: ", exp_args['num_layers_frozen'])
        train_accs, test_accs = transfer_learner.run_transfer_learner_reps(exp_args=exp_args)
        results_dict[l] = (train_accs, test_accs)
        print("=================================================================================================")
    print("layers".ljust(10), "train_mean".ljust(12), "train_std".ljust(12), "test_mean".ljust(12),
          "test_std".ljust(12))
    max_test_acc = -0.1
    max_lr = -0.1
    for lr in exp_args['layers']:
        train_mean = np.mean(np.asarray(results_dict[lr][0]))
        train_sd = np.std(np.asarray(results_dict[lr][0]))
        test_mean = np.mean(np.asarray(results_dict[lr][1]))
        test_sd = np.std(np.asarray(results_dict[lr][1]))
        print('{0:<10.0f} {1:<12.2f} {2:<12.2f} {3:<12.2f} {4:<12.2f}'.format(lr, train_mean, train_sd, test_mean,
                                                                              test_sd))
        if test_mean > max_test_acc:
            max_lr = lr
            max_test_acc = test_mean
    print("We got max test acc. of {0:.2f} with lr {1:.5f}".format(max_test_acc, max_lr))
    print("=================================================================================================")


if __name__ == "__main__":
    args = vars(get_parser(""))
    if args['type'] == TR_LRNR_HT:
        run_transfer_ht(exp_args=args)
    elif args['type'] == TR_LRNR_EV:
        run_transfer_ev(exp_args=args)
