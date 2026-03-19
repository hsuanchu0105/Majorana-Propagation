import numpy as np


def ErrorPrint(_dexp, _obexp, _rexp, _ord, logger=print):

    print('\t')
    print("-----------------Direct exponential---------------")
    print(_dexp)
    print("-----------------Majorana Propagation-------------")
    print(_obexp)
    print("------------Rotated Majorana Propagation----------")
    print(_rexp)



    #eps = 1e-15
    rel_maj = np.abs(_obexp - _dexp) / np.abs(_dexp)
    rel_rotm = np.abs(_rexp - _dexp) / np.abs(_dexp)

    print('\t')
    print("Relative error Majorana Propagation")
    print(rel_maj)

    print("Relative error rotated Majorana")
    print(rel_rotm)

    #2-norm 
    rel_ob_global = np.linalg.norm(_obexp - _dexp, ord = _ord) / np.linalg.norm(_dexp, ord = _ord)
    rel_re_global = np.linalg.norm(_rexp - _dexp, ord = _ord) / np.linalg.norm(_dexp, ord = _ord)

    logger('\t')
    logger("Global relative error")
    logger(rel_ob_global, rel_re_global)