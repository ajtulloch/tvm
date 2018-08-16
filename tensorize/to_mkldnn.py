import collections

Workload = collections.namedtuple("Workload", ["space", "input_channel", "output_channel", "kernel", "pad", "stride"])

WORKLOADS = [
    # Workload(space=224, input_channel=3, output_channel=64, kernel=3),
    Workload(space=56, input_channel=64, output_channel=64, kernel=3, pad=1, stride=1),
    Workload(space=56, input_channel=64, output_channel=128, kernel=3, pad=1, stride=1),
    Workload(space=28, input_channel=128, output_channel=128, kernel=3, pad=1, stride=1),
    Workload(space=28, input_channel=128, output_channel=256, kernel=3, pad=1, stride=1),
    Workload(space=14, input_channel=256, output_channel=256, kernel=3, pad=1, stride=1),
    Workload(space=14, input_channel=256, output_channel=512, kernel=3, pad=1, stride=1),
    Workload(space=7, input_channel=512, output_channel=512, kernel=3, pad=1, stride=1),

]
# WORKLOADS = [
#         # Workload(space=102, input_channel=128, output_channel=128, kernel=3, pad=1, stride=1),
#         # # Workload(space=102, input_channel=32, output_channel=32, kernel=3, pad=1, stride=1),
#         # # Workload(space=56, input_channel=64, output_channel=64, kernel=3, pad=1, stride=1),
#         # # Workload(space=56, input_channel=128, output_channel=128, kernel=3, pad=1, stride=1),
#         # # Workload(space=56, input_channel=256, output_channel=256, kernel=3, pad=1, stride=1),
#         # # Workload(space=56, input_channel=128, output_channel=128, kernel=3, pad=1, stride=1),
#         # # Workload(space=56, input_channel=256, output_channel=256, kernel=3, pad=1, stride=1),
#         # Workload(space=128, input_channel=64, output_channel=64, kernel=3, pad=1, stride=1),
#         # Workload(space=56, input_channel=256, output_channel=256, kernel=3, pad=1, stride=1),

#         # # Workload(space=12, input_channel=256, output_channel=256, kernel=3, pad=1, stride=1),
#         Workload(space=192, input_channel=3, output_channel=12, kernel=3, pad=1, stride=1),
#         Workload(space=96, input_channel=12, output_channel=24, kernel=3, pad=1, stride=1),
#         Workload(space=48, input_channel=24, output_channel=48, kernel=3, pad=1, stride=1),
#         Workload(space=24, input_channel=48, output_channel=96, kernel=3, pad=1, stride=1),
#         Workload(space=12, input_channel=96, output_channel=180, kernel=3, pad=1, stride=1),
#         Workload(space=6, input_channel=180, output_channel=220, kernel=3, pad=1, stride=1),
#         Workload(space=6, input_channel=220, output_channel=180, kernel=3, pad=1, stride=1),
#         Workload(space=12, input_channel=180, output_channel=96, kernel=3, pad=1, stride=1),
#         Workload(space=24, input_channel=96, output_channel=48, kernel=3, pad=1, stride=1),
#         Workload(space=48, input_channel=48, output_channel=24, kernel=3, pad=1, stride=1),
#         Workload(space=96, input_channel=24, output_channel=12, kernel=3, pad=1, stride=1),
#         Workload(space=192, input_channel=12, output_channel=1, kernel=3, pad=1, stride=1),
# ]

def format_workload(w, i):
    return 'mb1_g1_ic{w.input_channel}_oc{w.output_channel}_ih{w.space}_iw{w.space}_kh{w.kernel}_kw{w.kernel}_sh{w.stride}_sw{w.stride}_ph{w.pad}_pw{w.pad}_n"seg{i}"'.format(w=w, i=i)

for i, w in enumerate(WORKLOADS):
    print(format_workload(w, i))

import tempfile
import subprocess

def run_mkl(w, i):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(format_workload(w, i))
        f.close()
        print(f.name)
        try:
            subprocess.check_call(
                ["/Users/tulloch/src/mkl-dnn/build/tests/benchdnn/benchdnn",
                 "--mode=p",
                 "--conv",
                 "--cfg=f32",
                 "--dir=FWD_I",
                 "--alg=WINO",
                 "--batch={}".format(f.name)
                ])
        except:
            pass
            # print("Failed")

def run_nnpack(w, i):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(format_workload(w, i))
        f.close()
        subprocess.check_call(
            ["/Users/tulloch/src/NNPACK/build/bench-convolution-inference",
             "-ic",
             str(w.input_channel),
             "-oc",
             str(w.output_channel),
             "-is",
             str(w.space),
             str(w.space),
             "-ks",
             str(w.kernel),
             str(w.kernel),
             "-m",
             "inference",
             "-a",
             "wt8x8",
             "-b",
             "1",
             "-t",
             "0",
             "-ip",
             "1",
             "-ts",
             "precompute",
             "-i",
             "50"
            ])

for i, w in enumerate(WORKLOADS):
    print(run_mkl(w, i))
    # print(run_nnpack(w, i))
