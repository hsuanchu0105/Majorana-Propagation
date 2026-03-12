import numpy as np


def ErrorPrint(_dexp, _obexp, _rexp, _ord, logger=print):

    logger('\t')
    logger("-----------------Direct exponential---------------")
    logger(_dexp)
    logger("-----------------Majorana Propagation-------------")
    logger(_obexp)
    logger("------------Rotated Majorana Propagation----------")
    logger(_rexp)



    #eps = 1e-15
    rel_maj = np.abs(_obexp - _dexp) / np.abs(_dexp)
    rel_rotm = np.abs(_rexp - _dexp) / np.abs(_dexp)

    logger('\t')
    logger("Relative error Majorana Propagation")
    logger(rel_maj)

    logger("Relative error rotated Majorana")
    logger(rel_rotm)

    #2-norm 
    rel_ob_global = np.linalg.norm(_obexp - _dexp, ord = _ord) / np.linalg.norm(_dexp, ord = _ord)
    rel_re_global = np.linalg.norm(_rexp - _dexp, ord = _ord) / np.linalg.norm(_dexp, ord = _ord)

    logger('\t')
    logger("Global relative error")
    logger(rel_ob_global, rel_re_global)