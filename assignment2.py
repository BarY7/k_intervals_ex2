#################################
# Your name:
#################################

from os.path import sameopenfile
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.nanfunctions import nanmedian
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        xs = np.random.random_sample(m)
        ys = [0 for i in range(0, m)]
        for i in range(0, m):
            if((xs[i] >= 0.2 and xs[i] <= 0.4) or (xs[i] >= 0.6 and xs[i] <= 0.8)):
                ys[i] = np.random.choice([0, 1], size=1, p=[0.9, 0.1])[0]
            else:
                ys[i] = np.random.choice([0, 1], size=1, p=[0.2, 0.8])[0]
        return np.array([[xs[i], ys[i]] for i in range(0, m)])

    def draw_sample_intervals(self, m, k):
        """
        Plots the data as asked in (a) i ii and iii.
        Input: m - an integer, the size of the data sample.
               k - an integer, the maximum number of intervals.

        Returns: None.
        """
        samples = self.sample_from_D(m)
        samples = samples[samples[:, 0].argsort()]
        plt.scatter(samples[:, 0], samples[:, 1])
        plt.xlim(-0.1, 1.1)
        plt.ylim(-0.1, 1.1)
        plt.axvline(x=0.2)
        plt.axvline(x=0.4)
        plt.axvline(x=0.6)
        plt.axvline(x=0.8)
        (inters, best_err) = intervals.find_best_interval(
            samples[:, 0], samples[:, 1], 3)
        flat_inters = []
        for i in range(k):
            flat_inters.append(inters[i][0])
            flat_inters.append(inters[i][1])
        for interval in inters:
            dots = np.linspace(interval[0], interval[1], num=1000)
            plt.plot(dots, [-0.1] * 1000, linewidth=5)
        plt.xticks(flat_inters)
        plt.show()
        return inters

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        m_array = []
        for m in range(m_first, m_last+step, step):
            m_array.append(m)
        m_array = np.array(m_array)
        avg_empirical_array = np.array(
            [0 for x in range(m_first, m_last+step, step)], dtype=np.float32)
        avg_true_array = np.array(
            [0 for x in range(m_first, m_last+step, step)], dtype=np.float32)
        for t in range(0, T):
            for m in range(m_first, m_last+step, step):
                sample_array = self.sample_from_D(m)
                sample_array = sample_array[sample_array[:, 0].argsort()]
                xs = sample_array[:, 0]
                ys = sample_array[:, 1]
                all_intervals = intervals.find_best_interval(xs, ys, k)[0]
                # calculate empirical err
                sum = 0
                for index in range(len(xs)):
                    is_in_any_interval = False
                    for interval in all_intervals:
                        if(self.is_in_interval(interval, xs[index])):
                            is_in_any_interval = True
                    if((is_in_any_interval and ys[index] == 0) or
                            (not is_in_any_interval and ys[index] == 1)):
                        sum = sum + 1
                empirical_err = sum/m
                true_err = self.calculate_error_from_intervals(all_intervals)
                avg_empirical_array[int(
                    (m - m_first)/step)] = avg_empirical_array[int((m - m_first)/step)] + empirical_err/T
                avg_true_array[int(
                    (m - m_first)/step)] = avg_true_array[int((m - m_first)/step)] + true_err/T
        line_emp, = plt.plot(m_array, avg_empirical_array)
        line_true, = plt.plot(m_array, avg_true_array)
        plt.xlabel("m")
        plt.ylabel("error")
        plt.legend([line_emp, line_true], ['Empirical Error', 'True Error'])
        plt.show()

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        k_array, empirical_array, true_array = self.compute_errors_and_ks(
            m, k_first, k_last, step)
        line_emp, = plt.plot(k_array, empirical_array)
        line_true, = plt.plot(k_array, true_array)
        plt.xlabel("k")
        plt.ylabel("error")
        plt.legend([line_emp, line_true], ['Empirical Error', 'True Error'])
        plt.show()

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Runs the experiment in (d).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        k_array, empirical_array, true_array, penalty_array, penalty_emp_sum_array = self.compute_errors_and_ks_srm(
            m, k_first, k_last, step)
        line_emp, = plt.plot(k_array, empirical_array)
        line_true, = plt.plot(k_array, true_array)
        line_penalty, = plt.plot(k_array, penalty_array)
        line_emp_penalty_sum, = plt.plot(k_array, penalty_emp_sum_array)
        plt.xlabel("k")
        plt.ylabel("value")
        plt.legend([line_emp, line_true, line_penalty, line_emp_penalty_sum], [
                   'Empirical Error', 'True Error', 'Penalty', 'Penalty+Empirical'])
        plt.show()

    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
        Chooses the best hypothesis based on 3 experiments.
        Input: m - an integer, the size of the data sample.
               T - an integer, the number of times the experiment is performed.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        k_first = 1
        k_last = 10
        step = 1
        k_array = []
        for k in range(k_first, k_last+step, step):
            k_array.append(k)
        k_array = np.array(k_array)
        result = []
        for t in range(0, T):
            empirical_array = np.array(
                [0 for x in range(k_first, k_last+step, step)], dtype=np.float32)
            intervals_for_k = [0 for x in range(k_first, k_last+step, step)]
            sample_array_full = self.sample_from_D(m)
            for k in range(k_first, k_last+step, step):
                sample_array = sample_array_full[:int(0.8*m)]
                sample_array = sample_array[sample_array[:, 0].argsort()]
                xs = sample_array[:, 0]
                ys = sample_array[:, 1]
                all_intervals = intervals.find_best_interval(xs, ys, k)[0]
                # empirical_err,true_err = self.calculate_errors(sample_array,k,all_intervals)
                # empirical_array[int(
                #     (k - k_first)/step)] = empirical_array[int((k - k_first)/step)] + empirical_err
                intervals_for_k[int(
                    (k - k_first)/step)] = all_intervals
            for k in range(k_first, k_last+step, step):
                sample_array = sample_array_full[int(0.8*m):]
                sample_array = sample_array[sample_array[:, 1].argsort()]
                empirical_err, true_err = self.calculate_errors(
                    sample_array, k, intervals_for_k[int((k - k_first)/step)])
                empirical_array[int(
                    (k - k_first)/step)] = empirical_array[int((k - k_first)/step)] + empirical_err
            result.append(np.argmin(empirical_array) + 1)
        print("OPTIMAL SOLUTION is")
        print(result)
        return result,

    #################################
    # Place for additional methods

    def calculate_errors(self, sample_array, k, all_intervals=[]):
        xs = sample_array[:, 0]
        ys = sample_array[:, 1]
        if(len(all_intervals) == 0):
            all_intervals = intervals.find_best_interval(xs, ys, k)[0]
        # calculate empirical err
        sum = 0
        for index in range(len(xs)):
            is_in_any_interval = False
            for interval in all_intervals:
                if(self.is_in_interval(interval, xs[index])):
                    is_in_any_interval = True
            if((is_in_any_interval and ys[index] == 0) or
                    (not is_in_any_interval and ys[index] == 1)):
                sum = sum + 1
        empirical_err = sum/len(sample_array)
        true_err = self.calculate_error_from_intervals(all_intervals)
        return empirical_err, true_err
    # check if value is in interval

    def is_in_interval(self, interval, value):
        if(interval[0] <= value and interval[1] >= value):
            return True
        return False

    def calculate_intervals_intersection(self, int_a, int_b):
        if(int_a[1] < int_b[0] or int_b[1] < int_a[0]):
            return 0
        if(int_a[0] <= int_b[0] and int_a[1] >= int_b[1]):
            return int_b[1] - int_b[0]
        if(int_b[0] < int_a[0] and int_b[1] > int_a[1]):
            return int_a[1] - int_a[0]
        if(int_a[0] < int_b[0] and int_a[1] < int_b[1]):
            return int_a[1] - int_b[0]
        if(int_b[0] < int_a[0] and int_b[1] < int_a[1]):
            return int_b[1] - int_a[0]

    def calculate_error_from_intervals(self, list_of_1_intervals):
        p_intervals = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1)]
        axis_pointer = 0
        err_sum = 0
        for p_interval in p_intervals:
            total_for_interval = 0
            for x in list_of_1_intervals:
                total_for_interval = total_for_interval + \
                    self.calculate_intervals_intersection(p_interval, x)
            total_for_interval = total_for_interval / \
                (p_interval[1] - p_interval[0])
            if (p_interval == (0.2, 0.4) or p_interval == (0.6, 0.8)):
                err_sum = err_sum + (total_for_interval *
                                     0.18 + (1-total_for_interval)*0.02)
            else:
                err_sum = err_sum + (total_for_interval *
                                     0.04 + (1-total_for_interval)*0.16)
        return err_sum

    def calculate_penalty(self, n, k, delta):
        first = 8/n
        first_ln = np.log(40)
        second_ln = np.log((np.e * n)/(k))
        return np.sqrt(first*(first_ln + 2*k*second_ln))

    def compute_errors_and_ks(self, m, k_first, k_last, step):
        k_array = []
        for k in range(k_first, k_last+step, step):
            k_array.append(k)
        k_array = np.array(k_array)
        empirical_array = np.array(
            [0 for x in range(k_first, k_last+step, step)], dtype=np.float32)
        true_array = np.array(
            [0 for x in range(k_first, k_last+step, step)], dtype=np.float32)
        sample_array = self.sample_from_D(m)
        sample_array = sample_array[sample_array[:, 0].argsort()]
        for k in range(k_first, k_last+step, step):
            xs = sample_array[:, 0]
            ys = sample_array[:, 1]
            all_intervals = intervals.find_best_interval(xs, ys, k)[0]
            # calculate empirical err
            sum = 0
            for index in range(len(xs)):
                is_in_any_interval = False
                for interval in all_intervals:
                    if(self.is_in_interval(interval, xs[index])):
                        is_in_any_interval = True
                if((is_in_any_interval and ys[index] == 0) or
                        (not is_in_any_interval and ys[index] == 1)):
                    sum = sum + 1
            empirical_err = sum/m
            true_err = self.calculate_error_from_intervals(all_intervals)
            empirical_array[int(
                (k - k_first)/step)] = empirical_array[int((k - k_first)/step)] + empirical_err
            true_array[int(
                (k - k_first)/step)] = true_array[int((k - k_first)/step)] + true_err
        return k_array, empirical_array, true_array

    def compute_errors_and_ks_srm(self, m, k_first, k_last, step):
        k_array = []
        for k in range(k_first, k_last+step, step):
            k_array.append(k)
        k_array = np.array(k_array)
        empirical_array = np.array(
            [0 for x in range(k_first, k_last+step, step)], dtype=np.float32)
        true_array = np.array(
            [0 for x in range(k_first, k_last+step, step)], dtype=np.float32)
        penalty_array = np.array(
            [0 for x in range(k_first, k_last+step, step)], dtype=np.float32)
        penalty_emp_sum_array = np.array(
            [0 for x in range(k_first, k_last+step, step)], dtype=np.float32)
        sample_array = self.sample_from_D(m)
        sample_array = sample_array[sample_array[:, 0].argsort()]
        for k in range(k_first, k_last+step, step):
            xs = sample_array[:, 0]
            ys = sample_array[:, 1]
            all_intervals = intervals.find_best_interval(xs, ys, k)[0]
            # calculate empirical err
            sum = 0
            for index in range(len(xs)):
                is_in_any_interval = False
                for interval in all_intervals:
                    if(self.is_in_interval(interval, xs[index])):
                        is_in_any_interval = True
                if((is_in_any_interval and ys[index] == 0) or
                        (not is_in_any_interval and ys[index] == 1)):
                    sum = sum + 1
            empirical_err = sum/m
            true_err = self.calculate_error_from_intervals(all_intervals)
            penalty = self.calculate_penalty(m, k, 0.1)
            err_penalty_sum = penalty + empirical_err
            empirical_array[int(
                (k - k_first)/step)] = empirical_array[int((k - k_first)/step)] + empirical_err
            true_array[int(
                (k - k_first)/step)] = true_array[int((k - k_first)/step)] + true_err
            penalty_array[int(
                (k - k_first)/step)] = penalty_array[int((k - k_first)/step)] + penalty
            penalty_emp_sum_array[int(
                (k - k_first)/step)] = penalty_emp_sum_array[int((k - k_first)/step)] + err_penalty_sum
        return k_array, empirical_array, true_array, penalty_array, penalty_emp_sum_array

    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.draw_sample_intervals(100, 3)
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500, 3)
