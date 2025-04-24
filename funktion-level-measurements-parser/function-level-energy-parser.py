import os
import sys, getopt
import time
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
import seaborn as sns
import numpy as np
import subprocess
from datetime import datetime
from joblib import Parallel, delayed
import argparse


########################################################################################################################
# Parse Energy
########################################################################################################################

def parse_args(args):
    args_arr = literal_eval(args[0][6:-3])
    return [args_arr[1][args_arr[1].rfind('/') + 1:], args_arr[3]]


def string_to_topic(string):
    string = string[string.find(',') + 1:]
    if len(string) < 1:
        raise Exception('Could not extract topic.')
    return string.split('/')


def energy_value_frequency(line):
    end_measurements = line.find(']"')
    time_energy_vals = line[:end_measurements + 1]

    # print('Test')
    # print(time_energy_vals)

    try:
        time_energy_vals_arr = literal_eval(time_energy_vals)
        # print(time_energy_vals_arr)
    except SyntaxError:
        print('Error evaluating line: {}; num energy vals: {}'.format(line, len(time_energy_vals)))
        print('idx end measurements: {}; time_energy_vals: {}'.format(end_measurements, time_energy_vals))
        return None

    # t_start = time_energy_vals_arr[0] # start timestamp
    # t_end = time_energy_vals_arr[-1] # end timestamp

    t_start_pi = time_energy_vals_arr[1]  # pi
    energy_vals = time_energy_vals_arr[2:-1]
    if any(val < 0 for val in energy_vals):
        raise Exception('Energy value is negative in line:', line)
    return (t_start_pi, energy_vals)


def parse_energy(energy_vals):
    num_lines = len(energy_vals)
    df_container = []

    for i, line in enumerate(energy_vals):
        if line.startswith(',0,1'):
            continue
        else:
            topic_end = line.find(',"')
            topics = string_to_topic(line[:topic_end])
            try:
                start_time, energy = energy_value_frequency(line[topic_end + 2:])
            except TypeError:
                print('Could not extract energy values for line [{}]: {}'.format(i, line))
                sys.exit()

            energy_df = pd.DataFrame(data=energy, columns=['power'])
            energy_df['mqtt_Starttime'] = start_time
            energy_df['component'] = topics[2]
            energy_df['mqtt_M'] = i
            df_container.append(energy_df)

    return pd.concat(df_container, ignore_index=True)


def rowFunc(row, time_delta):
    return row['mqtt_Starttime'] + row['rowIndex'] * time_delta


def interpolate_mqtt_time(energy):
    df_arr = []
    for key_comp, df_comp in energy.groupby(['component']):
        df_comp = df_comp.reset_index(drop=True)
        mqtt_timestamps = df_comp.mqtt_Starttime.unique()
        # mqtt_time_deltas = []
        # for i in range(0, len(mqtt_timestamps) - 1):
        #     t0 = mqtt_timestamps[i]
        #     t1 = mqtt_timestamps[i + 1]
        #     mqtt_time_deltas.append((t1 - t0) / 799)
        # mqtt_time_deltas.append(mqtt_time_deltas[-1])
        # get time difference between MQTT messages for this component
        n_measurements_in_one_mqtt_package = 799
        mqtt_package_time_deltas = (mqtt_timestamps[1:] - mqtt_timestamps[:-1])
        avg_measurement_time_deltas_per_mqtt_package_fast = list(
            mqtt_package_time_deltas / n_measurements_in_one_mqtt_package)
        avg_measurement_time_deltas_per_mqtt_package_fast.append(avg_measurement_time_deltas_per_mqtt_package_fast[-1])
        mqtt_M_df_arr = []
        for pkg_time_delta, (mqtt_M_idx, out) in zip(avg_measurement_time_deltas_per_mqtt_package_fast,
                                                     enumerate(df_comp.groupby(['mqtt_M']))):
            _, mqtt_M_df = out
            current_delta = avg_measurement_time_deltas_per_mqtt_package_fast[mqtt_M_idx]

            cum_deltas = np.arange(0, n_measurements_in_one_mqtt_package) * current_delta
            # TODO check why first delta should have no effect as it is multiplied with 0
            mqtt_Starttime = mqtt_M_df.iloc[0]['mqtt_Starttime']
            mqtt_M_df['mqtt_Time'] = mqtt_Starttime + cum_deltas
            # starttimestamp
            # mqtt_M_df = mqtt_M_df.reset_index(drop=True)
            # mqtt_M_df['rowIndex'] = mqtt_M_df.index
            # mqtt_M_df['mqtt_Time'] = mqtt_M_df.apply(rowFunc, time_delta=current_delta, axis=1)
            mqtt_M_df_arr.append(mqtt_M_df)

        df_arr.append(pd.concat(mqtt_M_df_arr))
    return pd.concat(df_arr).reset_index(drop=True)


########################################################################################################################
# Parse Performance
########################################################################################################################


def parse_sample_start(line):
    #        perf 325851 2145171.658899497:          1 cycles:
    # parts: perf_identifier, thread_id, timestamp:, num_cycles, cycles_identifier

    parts = line.split()
    return (parts[1], float(parts[2][:-1]))


def parse_call_stack(stack):
    for stack_trace in stack:
        #        ffffffff91478b26 native_write_msr+0x6 ([kernel.kallsyms])
        # parts: stack_trace hash, funkction_name?, function_path?
        if 'x264' in stack_trace:
            x264_elements = stack_trace.split()
            return x264_elements[1].split('+')[0]

        #        56151aee75c4 do_primes_t_200000+0x10f (/tmp/mweber/experiment/artificial_gen.out)
        # parts: stack_trace hash, funkction_name?, function_path?
        elif 'artif' in stack_trace:
            artif_elements = stack_trace.split()
            return artif_elements[1].split('+')[0]

    stack_top = stack[0].split()
    # stack_top[1].split('+')[0]
    return stack_top[1].split('+')[0]


def get_methods_streak(df):
    method_streak = []

    current_streak = 0
    last_method = ''
    start_timestamp = 0
    current_row = None

    # input_medtods_name = df['method'].to_numpy()
    # input_methods_time = df['exp_time'].to_numpy()
    # input_methods_idx = 0

    # while input_methods_idx < len(df):
    for index, row in df.iterrows():

        current_row = row
        if row['method'] == last_method:
            # if input_medtods_name
            current_streak += 1
        else:
            method_streak.append([last_method, current_streak, start_timestamp, row['exp_time']])
            last_method = row['method']
            current_streak = 1
            start_timestamp = row['exp_time']

    # append also last method
    method_streak.append([last_method, current_streak, start_timestamp,
                          current_row['exp_time'] + (current_row['exp_time'] - start_timestamp)])

    return pd.DataFrame(data=method_streak[1:], columns=['method', 'streak', 't_start', 't_end'])


def extract_profile(profile):
    call_stack = []

    thread_id = 0
    timestamp = 0
    profile_out = []

    for line in profile:
        line = line.strip()

        if (line.startswith('perf ')) | (line.startswith('x264 ')) | (line.startswith('artificial_gen')) | (
                line.startswith('lrzip')) | (line.startswith('brotli')):
            thread_id, timestamp = parse_sample_start(line)
            call_stack = []

        elif len(line) == 0:
            if call_stack:
                top_stack_method = parse_call_stack(call_stack)
                profile_out.append([thread_id, timestamp, top_stack_method])
                call_stack = []
            else:
                print('Performance stack empty at timestamp {}'.format(timestamp))

        else:
            call_stack.append(line)

    return pd.DataFrame(data=profile_out, columns=['thread_id', 'timestamp', 'method'])


########################################################################################################################
# Unite Performance and Energy
########################################################################################################################


def interpolate_energy_m(t1, p1, t2, p2, t):
    if t == t1:
        return p1
    elif t == t2:
        return p2

    if not ((t1 < t) & (t2 > t)):
        raise ValueError('Current t {} is not in between {}, {}.'.format(t, t1, t2))

    p_t = p1 + ((t - t1) * ((p2 - p1) / (t2 - t1)))
    return p_t


def calc_component_method_energy_slow(energy_df_comp, tmp_method_streak_df_2, performance_start):
    measurements_time_values = energy_df_comp['mqtt_Time'].to_numpy()
    measurements_power_values = energy_df_comp['power'].to_numpy()
    energy_measurements_idx = 0

    iterator = energy_df_comp.iterrows()
    _, last = next(iterator)

    method_energy_values = []

    collected_t = []
    collected_p = []

    n_method_calls = len(tmp_method_streak_df_2)
    all_method_starts = tmp_method_streak_df_2['t_start'].to_numpy()
    all_method_ends = tmp_method_streak_df_2['t_end'].to_numpy()

    current_method_idx = 0

    while True:
        energy_measurements_idx += 1
        current_energy_time = measurements_time_values[energy_measurements_idx]

        # 1
        if current_energy_time < performance_start:
            continue

        start_current_method = all_method_starts[current_method_idx]
        end_current_method = all_method_ends[current_method_idx]

        t1 = measurements_time_values[energy_measurements_idx - 1]
        t2 = measurements_time_values[energy_measurements_idx]
        p1 = measurements_power_values[energy_measurements_idx - 1]
        p2 = measurements_power_values[energy_measurements_idx]

        curr_p = interpolate_energy_m(t1, p1, t2, p2, start_current_method)
        collected_t.append(start_current_method)
        collected_p.append(curr_p)

        collected_t.append(t2)
        collected_p.append(p2)
        break

    while energy_measurements_idx < len(measurements_time_values) - 1:
        energy_measurements_idx += 1
        current_energy_time = measurements_time_values[energy_measurements_idx]

        start_current_method = all_method_starts[current_method_idx]
        end_current_method = all_method_ends[current_method_idx]

        t1 = measurements_time_values[energy_measurements_idx - 1]
        t2 = measurements_time_values[energy_measurements_idx]
        p1 = measurements_power_values[energy_measurements_idx - 1]
        p2 = measurements_power_values[energy_measurements_idx]

        if current_energy_time < end_current_method:
            collected_t.append(t2)
            collected_p.append(p2)
        else:
            curr_p = interpolate_energy_m(t1, p1, t2, p2, end_current_method)
            collected_t.append(end_current_method)
            collected_p.append(curr_p)

            energy_m = np.trapz(x=collected_t, y=collected_p)
            if energy_m < 0:
                raise Exception('Energy consumption of method {} can not be negative.'.format(energy_measurements_idx),
                                energy_m, current_energy_time)
            method_energy_values.append(energy_m)
            # print('Total energy of method {}'.format(current_method['method']), energy_m)

            collected_t = [end_current_method]
            collected_p = [curr_p]

            # t2 ~ end_current_method
            while True:
                current_method_idx += 1
                if current_method_idx == n_method_calls:
                    return method_energy_values

                start_current_method = all_method_starts[current_method_idx]
                end_current_method = all_method_ends[current_method_idx]

                if end_current_method < t2:
                    curr_p = interpolate_energy_m(t1, p1, t2, p2, end_current_method)

                    collected_t.append(end_current_method)
                    collected_p.append(curr_p)
                    energy_m = np.trapz(x=collected_t, y=collected_p)

                    method_energy_values.append(energy_m)
                    # print('Total energy of method {}'.format(current_method['method']), energy_m)

                    collected_t = [end_current_method]
                    collected_p = [curr_p]

                else:
                    collected_t.append(t2)
                    collected_p.append(p2)
                    break

        # last = current
    return method_energy_values


def extract_start_time(path):
    time_search_string = '[WL] end cooldown time: '
    log_path = os.path.dirname(path)
    files = os.listdir(log_path)
    for file in files:
        if not file.endswith('.out'):
            continue
        with open(os.path.join(log_path, file)) as f:
            for line in f.readlines():
                if line.startswith(time_search_string):
                    return float(line[len(time_search_string):].strip())
    return None


def extract_experiment(path):
    parts = path.split('/')
    return (parts[-3], parts[-2], parts[-1])


def parallel_unite(component, comp_df, cp_df, performance_start):
    print('Current component:', component)
    cp_df['energy'] = calc_component_method_energy_slow(comp_df, cp_df, performance_start)
    cp_df['component'] = component
    return cp_df


########################################################################################################################
# Unite Performance and Energy
########################################################################################################################

def merge_anyway(df, arr, identifier):
    if len(df) > len(arr):
        df = df.drop(index=df.index[:len(arr)], axis=0, inplace=True)
        df[identifier] = arr
        return df
    else:
        arr = arr[0:len(df)]
        df[identifier] = arr
        return df


def parallel_experiment_data_extraction(path, results_path, slow_but_steady_computations=False, vectorized_computation = True):

    if not slow_but_steady_computations and not vectorized_computation:
        vectorized_computation = True

    print('starting', path)
    file = os.path.join(path, 'energy_measurements.csv')
    parsing_t_start = time.time()
    energy_tmp = get_energy_mqtt_times(file)

    # end energy measurement data parsing
    # start performance measurement data parsing
    perf_path_sub = os.path.dirname(os.path.dirname(path))
    perf_path = os.path.join(perf_path_sub, 'performance')
    perf_profile_df = get_performance_profile(perf_path)

    # end performance data
    # merge energy and performance:
    # print('start merging energy and performance', file)
    performance_start = perf_profile_df['t_start'].min()
    comp_wise_energy = []

    t_start_energy_integration = time.time()
    for component, comp_df in energy_tmp.groupby('component'):
        # print('Current component:', component)
        cp_df = perf_profile_df.copy()
        if slow_but_steady_computations:
            out = calc_component_method_energy_slow(comp_df, perf_profile_df, performance_start)
            consumptions = out
        if vectorized_computation:
            vectorized_consumptions = get_vectorized_energy_consumption_per_method_call(comp_df, perf_profile_df)
            consumptions = vectorized_consumptions
        if len(consumptions) != len(cp_df):
            print('Length of energy data do not match performance data: {}'.format(path))
            cp_df['energy'] = -1
        else:
            cp_df['energy'] = consumptions
        cp_df['component'] = component
        comp_wise_energy.append(cp_df)

    #print(f"Energy integration took {time.time() - t_start_energy_integration}s")

    out_df = pd.concat(comp_wise_energy)
    machine, jobID, taskID = extract_experiment(perf_path_sub)
    out_df['machine'] = machine
    out_df['jobID'] = jobID
    out_df['taskID'] = taskID
    parsing_time = time.time() - parsing_t_start
    #print('Time for parsing experiment:', parsing_time, path)

    out_filename = os.path.join(results_path, taskID+'.csv')
    out_df.to_csv(out_filename)


def get_performance_profile(perf_path):
    # performance start time
    t_performance = extract_start_time(perf_path)
    # performance_start_time[file] = t_performance
    # performance data
    file = os.path.join(perf_path, 'out.perf')
    with open(file) as f:
        profile = f.readlines()
        df = extract_profile(profile)
        df['exp_time_rel'] = df["timestamp"] - min(df["timestamp"])
        df['exp_time'] = df['exp_time_rel'] + t_performance

        perf_tmp = get_methods_streak(df)
        perf_tmp['t_delta'] = perf_tmp['t_end'] - perf_tmp['t_start']
    return perf_tmp


def get_energy_mqtt_times(file):
    with open(file) as f:
        start_reading = time.time()
        data = f.readlines()
        #print(f"Reading measurements took {time.time() - start_reading}s")
        out_df = parse_energy(data)
        start_time_conversion = time.time()
        energy_tmp = interpolate_mqtt_time(out_df)
        #print(f"Converting MQTT times took {time.time() - start_time_conversion}s")
        energy_tmp['exp_time'] = energy_tmp["mqtt_Time"] - min(energy_tmp["mqtt_Time"])
        # energy_data.append(energy_tmp)
    return energy_tmp


def get_vectorized_energy_consumption_per_method_call(comp_df, perf_tmp):
    ###  NEW METHOD
    power_measured = comp_df["power"]
    t_measurements = comp_df["mqtt_Time"]
    n_measurements = len(t_measurements)
    t_method_start = perf_tmp.t_start
    t_method_end = perf_tmp.t_end
    n_method_calls = len(t_method_start)
    # TODO there are methods that start on the same timestamp!
    power_method_start = np.interp(xp=t_measurements, fp=power_measured, x=t_method_start)
    power_method_end = np.interp(xp=t_measurements, fp=power_measured, x=t_method_end)
    all_times = np.array([*t_measurements, *t_method_start, *t_method_end])
    # get list that carries indexes of sorted elements in original list where indexes refer to unsorted list
    sorted_times_idx_forward = np.argsort(all_times)
    # get list that carries indexes from sorted index that gives the position in the sorted list
    sorted_times_idx = np.zeros(len(all_times))
    sorted_times_idx[sorted_times_idx_forward] = np.arange(len(all_times))
    sorted_times_idx = sorted_times_idx.astype(int)
    idx_measurements = sorted_times_idx[:n_measurements]
    idx_method_starts = sorted_times_idx[n_measurements:n_measurements + n_method_calls]
    idx_method_ends = sorted_times_idx[n_measurements + n_method_calls:]
    times_sorted = all_times[sorted_times_idx_forward]
    all_powers = np.array([*power_measured, *power_method_start, *power_method_end])
    time_sorted_powers = all_powers[sorted_times_idx_forward]
    method_idx_slices = tuple([np.s_[i:j] for (i, j) in zip(idx_method_starts, idx_method_ends)])
    consumptions = []
    for method_slice in method_idx_slices:
        method_all_times = times_sorted[method_slice]
        method_all_powers = time_sorted_powers[method_slice]
        method_consumption = np.trapz(x=method_all_times, y=method_all_powers)
        consumptions.append(method_consumption)
    return consumptions


def execute_extraction(files, results_path):
    Parallel(n_jobs=20)(delayed(parallel_experiment_data_extraction)(file, results_path) for file in files)


def serial_extraction(files, results_path):
    for file in files:
        parallel_experiment_data_extraction(file, results_path)


def get_experiments(focus_exp, exp_path):
    print('Experiment to process is: {}'.format(focus_exp))
    configurations = []
    known_files = ['out.perf', 'parameter.txt', 'perf.data', '.out', '.err']

    for path, dirs, files in os.walk(exp_path):
        for filename in files:
            file = os.path.join(path, filename)

            if focus_exp not in file:
                continue

            if any(substring in filename for substring in known_files):
                pass

            elif 'energy_measurements.csv' == filename:
                configurations.append(path)

            else:
                print('Not specified file:', file)
    print('Found {} experiments.'.format(len(configurations)))
    return configurations


def check_arguments():
    parser = argparse.ArgumentParser(
        prog='FAMLEM data parser',
        description='Parses FAMLEM energy and performance measurements and aggregates them'
    )
    parser.add_argument('--measurements', type=str,
                        help='parent folder of the measurements (./.../curie0[1|2])', required=True)
    parser.add_argument('--out', type=str,
                        help='parent results folder for output', required=True)
    parser.add_argument('--focus', type=str,
                        help='Slurm TaskID of the experiment', required=True)
    parser.add_argument('--serial', help='makes parallel executions serial',
                        action='store_true')

    args = parser.parse_args()
    in_path = args.measurements
    out_path = args.out
    os.makedirs(out_path, exist_ok=True)
    focus_exp = args.focus
    parallel = not args.serial
    return in_path, out_path, focus_exp, parallel


def parse_raw_measurement_data(in_path, results_path, focus_exp, parallel):
    start_time = time.time()
    configurations = get_experiments(focus_exp, in_path)

    if parallel:
        execute_extraction(configurations, results_path)
    else:
        serial_extraction(configurations, results_path)

    parsing_time = time.time() - start_time
    return parsing_time


def sum_eng_perf(path, file):
    res_file = os.path.join(path, file)
    print('process {}'.format(res_file))
    df = pd.read_csv(res_file)
    compressed_df = df.groupby(['machine', 'jobID', 'taskID', 'component', 'method'])[['t_delta', 'energy']].sum()
    compressed_df = compressed_df.reset_index()
    return compressed_df


def concat_experiment(results_path, focus_exp, parallel):
    start_time = time.time()
    all_result_file_names = [f for f in os.listdir(results_path) if os.path.isfile(os.path.join(results_path, f))]
    all_result_file_names = sorted(all_result_file_names)
    all_dfs = []

    if parallel:
        all_dfs = Parallel(n_jobs=50)(delayed(sum_eng_perf)(results_path, filename)
                                      for filename in all_result_file_names)
    else:
        for filename in all_result_file_names:
            compressed_df = sum_eng_perf(filename, results_path)
            all_dfs.append(compressed_df)

    fin_df = pd.concat(all_dfs)
    fin_df = fin_df.reset_index()

    results_path = os.path.join(os.path.dirname(results_path), '{}_ml.csv'.format(focus_exp))
    fin_df.to_csv(results_path)

    concat_time = time.time() - start_time
    return concat_time


def main():
    in_path, out_path, focus_exp, parallel = check_arguments()
    results_path = os.path.join(out_path, focus_exp)
    os.makedirs(results_path, exist_ok=True)

    #parsing_time = parse_raw_measurement_data(in_path, results_path, focus_exp, parallel)
    #print('Time reading data: {}'.format(parsing_time))

    concat_time = concat_experiment(results_path, focus_exp, parallel)
    print('Time concatinating experiments: {}'.format(concat_time))


if __name__ == '__main__':
    main()
