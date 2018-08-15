import collections

Workload = collections.namedtuple("Workload", ["space", "input_channel", "output_channel", "kernel", "pad", "stride"])

WORKLOADS = [
        # Workload(space=102, input_channel=128, output_channel=128, kernel=3, pad=1, stride=1),
        # # Workload(space=102, input_channel=32, output_channel=32, kernel=3, pad=1, stride=1),
        # # Workload(space=56, input_channel=64, output_channel=64, kernel=3, pad=1, stride=1),
        # # Workload(space=56, input_channel=128, output_channel=128, kernel=3, pad=1, stride=1),
        # # Workload(space=56, input_channel=256, output_channel=256, kernel=3, pad=1, stride=1),
        # # Workload(space=56, input_channel=128, output_channel=128, kernel=3, pad=1, stride=1),
        # # Workload(space=56, input_channel=256, output_channel=256, kernel=3, pad=1, stride=1),
        # Workload(space=128, input_channel=64, output_channel=64, kernel=3, pad=1, stride=1),
        # Workload(space=56, input_channel=256, output_channel=256, kernel=3, pad=1, stride=1),

        # # Workload(space=12, input_channel=256, output_channel=256, kernel=3, pad=1, stride=1),
        Workload(space=192, input_channel=3, output_channel=12, kernel=3, pad=1, stride=1),
        Workload(space=96, input_channel=12, output_channel=24, kernel=3, pad=1, stride=1),
        Workload(space=48, input_channel=24, output_channel=48, kernel=3, pad=1, stride=1),
        Workload(space=24, input_channel=48, output_channel=96, kernel=3, pad=1, stride=1),
        Workload(space=12, input_channel=96, output_channel=180, kernel=3, pad=1, stride=1),
        Workload(space=6, input_channel=180, output_channel=220, kernel=3, pad=1, stride=1),
        Workload(space=6, input_channel=220, output_channel=180, kernel=3, pad=1, stride=1),
        Workload(space=12, input_channel=180, output_channel=96, kernel=3, pad=1, stride=1),
        Workload(space=24, input_channel=96, output_channel=48, kernel=3, pad=1, stride=1),
        Workload(space=48, input_channel=48, output_channel=24, kernel=3, pad=1, stride=1),
        Workload(space=96, input_channel=24, output_channel=12, kernel=3, pad=1, stride=1),
        Workload(space=192, input_channel=12, output_channel=1, kernel=3, pad=1, stride=1),
]

def format_workload(w, i):
    return 'mb1_g1_ic{w.input_channel}_oc{w.output_channel}_ih{w.space}_iw{w.space}_kh{w.kernel}_kw{w.kernel}_sh{w.stride}_sw{w.stride}_ph{w.pad}_pw{w.pad}_n"seg{i}"'.format(w=w, i=i)

for i, w in enumerate(WORKLOADS):
    print(format_workload(w, i))
