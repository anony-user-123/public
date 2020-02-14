from math import sqrt
import numpy as np
from numpy import log, transpose
from numpy.linalg import inv
from scipy.stats import norm
import time


def eta(T):
    """
    Generates the cutoff probabilities for exploration rounds in interval
    chaining.

    :param T: the total number of iterations
    """
    return np.array([pow(t, -1/3) for t in range(1, T+1)])


def beta(k, d, c):
    """
    Generates the scaled down feature weights for a true model from the
    distribution β ∼ U[0, c]^d.

    :param k: the number of arms
    :param d: the number of features
    :param c: the scale of the feature weights
    """
    return np.random.uniform(0, c+1, size=(k, d))


def print_progress(s, should_print):
    """
    Helper function to print the progress of an algorithm as it's running.

    :param s: the string to print
    :should_print: whether or not the string should be printed
    """
    if should_print:
        print(s)


def top_interval(X, Y, k, d, _delta, T, _print_progress=True):
    """
    Simulates T rounds of TopInterval for k.

    :param X: a 3-axis (T, k, d) ndarray of d-dimensional context vectors for
              each time-step and arm
    :param Y: a T x k ndarray of reward function output for each context vector
    :param k: the number of arms
    :param d: the number of features
    :param _delta: confidence parameter
    :param T: the number of iterations
    :param _print_progress: True if progress should be printed; False otherwise
    :returns: cum_regret (the total regret across all T runs of the algorithm),
              avg_regret (the regret averaged across all T runs of the algorithm),
              final_regret (the regret in the last round of the algorithm)
    """
    pp = _print_progress
    _eta = eta(T)  # exploration cutoff probabilities
    picks = []
    for t in range(T):
        print_progress('Iteration [{0} / {1}]'.format(t, T), pp)
        if t <= d or np.random.rand() <= _eta[t]:
            # Play uniformly at random from [1, k].
            picks.append(np.random.randint(0, k))
            print_progress('Exploration round.', pp)
        else:
            intervals = []
            for i in range(k):
                # Compute beta hat.
                _Xti = X[:t+1, i]
                _XtiT = transpose(_Xti)
                try:
                    _XTX = inv(_XtiT.dot(_Xti))
                except:
                    print_progress('Encountered singular matrix. Ignoring.', pp)
                    continue
                _Yti = Y[:t+1, i]
                Bh_t_i = _XTX.dot(_XtiT).dot(_Yti)  # Compute OLS estimators.
                yh_t_i = Bh_t_i.dot(X[t, i])
                _s2 = np.var(Y[:t+1, i])
                # Compute the confidence interval width using the inverse CDF.
                w_t_i = norm.ppf(1 - _delta/(2*T*k), loc=0,
                                 scale=np.sqrt(_s2 * X[t, i].dot(_XTX).dot(transpose(X[t, i]))))
                intervals.append([yh_t_i - w_t_i, yh_t_i + w_t_i])
            # Pick the agent with the largest upper bound.
            picks.append(np.argmax(np.array(intervals)[:, 1]) if intervals else np.random.randint(0, k))
            print_progress('Intervals: {0}'.format(intervals), pp)
    # Compute sum of best picks over each iteration.
    best = [Y[i].max() for i in range(2, T)]
    performance = [Y[t][picks[t-2]] for t in range(2, T)]
    cum_regret = sum(best) - sum(performance)
    avg_regret = cum_regret / float(T)
    final_regret = best[-1] - performance[-1]
    print_progress('Cumulative Regret: {0}'.format(cum_regret), pp)
    print_progress('Average Regret: {0}'.format(avg_regret), pp)
    print_progress('Final Regret: {0}'.format(final_regret), pp)
    return cum_regret, avg_regret, final_regret


def compute_chain(i_st, intervals, k):
    # Sort intervals by decreasing order.
    chain = [i_st]
    ordering = np.argsort(intervals[:,1])[::-1]
    intervals = intervals[ordering,:]
    
    lowest_in_chain = intervals[0][0]
    for i in range(len(intervals)):
        if intervals[i][1] >= lowest_in_chain:
            chain.append(i)
            lowest_in_chain = min(lowest_in_chain, intervals[i][0])
        else:
            return chain
    return chain


def interval_chaining(X, Y, c, k, d, _delta, T, _print_progress=True):
    """
    Simulates T rounds of TopInterval for k.

    :param X: a 3-axis (T, k, d) ndarray of d-dimensional context vectors for
              each time-step and arm
    :param Y: a T x k ndarray of reward function output for each context vector
    :param k: the number of arms
    :param d: the number of features
    :param _delta: confidence parameter
    :param T: the number of iterations
    :param _print_progress: True if progress should be printed; False otherwise
    :returns: cum_regret (the total regret across all T runs of the algorithm),
              avg_regret (the regret averaged across all T runs of the algorithm),
              final_regret (the regret in the last round of the algorithm)
    """
    pp = _print_progress
    _eta = eta(T)  # exploration cutoff probabilities
    picks = []
    for t in range(T):
        print_progress('Iteration [{0} / {1}]'.format(t, T), pp)
        if t <= d or np.random.rand() <= _eta[t]:
            # Play uniformly at random from [1, k].
            picks.append(np.random.randint(0, k))
            print_progress('Exploration round.', pp)
        else:
            intervals = []
            for i in range(k):
                # Compute beta hat.
                _Xti = X[:t+1, i]
                _XtiT = transpose(_Xti)
                try:
                    _XTX = inv(_XtiT.dot(_Xti))
                except:
                    print_progress('Encountered singular matrix. Ignoring.', pp)
                    continue
                _Yti = Y[:t+1, i]
                Bh_t_i = _XTX.dot(_XtiT).dot(_Yti)  # Compute OLS estimators.
                yh_t_i = Bh_t_i.dot(X[t, i])
                _s2 = np.var(Y[:t+1, i])
                # Compute the confidence interval width using the inverse CDF.
                w_t_i = norm.ppf(1 - _delta/(2*T*k), loc=0,
                                 scale=np.sqrt(_s2 * X[t, i].dot(_XTX).dot(transpose(X[t, i]))))
                intervals.append([yh_t_i - w_t_i, yh_t_i + w_t_i])
            # Pick the agent with the largest upper bound.
            if not intervals:
                picks.append(np.random.randint(0, k))
            else:
                i_st = np.argmax(np.array(intervals)[:, 1])

                # Chaining
                chain = compute_chain(i_st, np.array(intervals), k)
                print_progress('Computed chain: {0}'.format(chain), pp)
                picks.append(np.random.choice(chain))
            print_progress('Intervals: {0}'.format(intervals), pp)
    # Compute sum of best picks over each iteration.
    best = [Y[i].max() for i in range(2, T)]
    performance = [Y[t][picks[t-2]] for t in range(2, T)]
    cum_regret = sum(best) - sum(performance)
    avg_regret = cum_regret / float(T)
    final_regret = best[-1] - performance[-1]
    print_progress('Cumulative Regret: {0}'.format(cum_regret), pp)
    print_progress('Average Regret: {0}'.format(avg_regret), pp)
    print_progress('Final Regret: {0}'.format(final_regret), pp)
    return cum_regret, avg_regret, final_regret


def ridge_fair(X, Y, k, d, _delta, T, _lambda, val_freq, val_data,
               model_selection_cri, logger, evaluator,
               _print_progress=True):
    """
    Simulates T rounds of ridge_fair.

    :param X: a 3-axis (T, k, d) ndarray of d-dimensional context vectors for
              each time-step and arm
    :param Y: a T x k ndarray of reward function output for each context vector
    :param k: the number of arms
    :param d: the number of features
    :param _delta: confidence parameter
    :param T: the number of iterations
    :param _lambda: regularization paramameter
    """
    picks = []
    B_best = np.zeros((k, d))
    B = np.zeros((k, d))
    best_rslt = float('-inf')
    for t in range(T):
        tick = time.time()
        for i in range(k):
            R = 1
            intervals = []
            try:
                X_i = X[:t, i]  # design matrix
                Y_i = Y[:t, i]  # same with Y
                x_ti = X[t, i]  # feature vector for arm i in round t

                X_iT = transpose(X_i)
                _idenD = np.identity(d)
                V_it = X_iT.dot(X_i) + (_lambda * _idenD)

                B_it = inv(V_it).dot(X_iT).dot(Y_i)
                B[i, :] = B_it

                y_ti = transpose(x_ti).dot(B_it)

                V_itI = inv(V_it)  # inverse of V_it
                _wti1 = sqrt(transpose(x_ti).dot(V_itI).dot(x_ti))
                _wti2 = R * sqrt(d * log((1 + (t / _lambda)) / _delta)) + sqrt(_lambda)
                w_ti = _wti1 * _wti2

                intervals.append([y_ti - w_ti, y_ti + w_ti])
            except:
                    print_progress('Error in assigning interval value.', _print_progress)
                    intervals = None
                    break

            if not intervals:
                picks.append(np.random.randint(0, k))
            else:
                i_st = np.argmax(np.array(intervals)[:, 1])
                chain = compute_chain(i_st, np.array(intervals), k)
                # play uniformly random from chain
                picks.append(np.random.choice(chain))

        tock = time.time()
        if t % 100 == 0:
            print(tock - tick, t, T)
            
        if t % val_freq == 0 and t > 0:
            acc, discri = evaluator.evaluate(B, val_data, 
                                             eval_consistency=False)
            if model_selection_cri == 'discri':
                val_score = -discri
            elif model_selection_cri == 'delta':
                val_score = acc - discri
            if val_score > best_rslt:
                best_rslt = val_score
                B_best = np.array(B)
                log_info = 'Iteration: {} acc = {} discri = {}'.format(
                            t, acc, discri)
                logger.info(log_info)

    best = [Y[i].max() for i in range(2, T)]
    performance = [Y[t][picks[t - 2]] for t in range(2, T)]
    print_progress('Cumulative Regret: {0}'.format(sum(best) - sum(performance)), _print_progress)
    print_progress('Final Regret: {0}'.format(best[-1] - performance[-1]), _print_progress)
    return B_best, best_rslt
