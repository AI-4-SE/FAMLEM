import os.path
import time


def parse_configurations(configurations):
    pass


def get_configurations(measurement_data_path, job_id):
    job_id_path = os.path.join(measurement_data_path, job_id)
    slurm_array_ids = os.listdir(job_id_path)

    configurations = []

    i = 0
    for file in slurm_array_ids:
        d = os.path.join(job_id_path, file)
        if os.path.isdir(d):
            configurations.append(d)
            i += 1

    print('{} configurations to process'.format(i))
    return configurations


def main():
    start_time = time.time()

    measurement_data_path = '/home/mweber/measurements/curie01/'
    out_path = '/home/mweber/fame_eval/output/'
    job_id = '1263284'
    print('Experiment to process is: {}'.format(job_id))

    configurations_to_extract = get_configurations(measurement_data_path, job_id)
    parse_configurations(configurations_to_extract)


if __name__ == '__main__':
    main()
