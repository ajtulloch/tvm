import logging
import sys
import click
import tvm
import numpy
import timeit
from tvm.contrib import cblas
from tvm import autotvm
from topi.util import get_const_int, get_const_tuple

@autotvm.template
def mm_NCn(M, N, K, P, dtype):
    assert M % P == 0
    A = tvm.placeholder((M // P, K, P), name='A', dtype=dtype)
    B = tvm.placeholder((N, K), name='B', dtype=dtype)
    k = tvm.reduce_axis((0, K), name='k')
    C = tvm.compute(
        (M // P, N, P),
        lambda mm, n, mmm: tvm.sum(A[mm, k, mmm] * B[n, k], axis=k),
        tag='batch_matmul_packed'
    )

    s = tvm.create_schedule(C.op)
    mm, n, mmm = s[C].op.axis
    k, = s[C].op.reduce_axis

    cfg = autotvm.get_config()
    ko, ki = cfg.define_split('tile_k', k, num_outputs=2)
    no, ni = cfg.define_split('tile_n', n, num_outputs=2)
    mm_, mmm_ = cfg.axis(mm), cfg.axis(mmm)

    cfg.define_reorder(
        "reorder_0",
        [mm_, mmm_, no, ni, ko, ki],
        policy='candidate',
        candidate=[
            # [mm_, no, ni, ko, ki, mmm_],
            [mm_, no, ko, ni, ki, mmm_],
            [mm_, no, ni, ko, ki, mmm_],
            [mm_, ko, no, ni, ki, mmm_],
            [mm_, ko, no, ki, ni, mmm_],
            [mm_, no, ko, ki, ni, mmm_],
        ]
    )

    cfg.define_annotate("ann_spatial", [ni, ki], policy='try_unroll')
    ko, ki = cfg['tile_k'].apply(s, C, k)
    no, ni = cfg['tile_n'].apply(s, C, n)
    cfg['reorder_0'].apply(s, C, [mm, mmm, no, ni, ko, ki])
    cfg['ann_spatial'].apply(
        s,
        C,
        [ki, ni],
        axis_lens=[
            cfg['tile_k'].size[-1],
            cfg['tile_n'].size[-1]
        ],
        max_unroll=16,
        cfg=cfg,
    )
    s[C].vectorize(mmm)
    return s, [A, B, C]

TARGETS = dict(
    skl='llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu',
    local='llvm -mcpu=core-avx2'
)

dtype = "float32"

N = 80
M = 16
K = 160
P = 8

@click.command()
@click.option('--autotvm_number', default=50)
@click.option('--autotvm_repeat', default=4)
@click.option('--autotvm_n_trial', default=200)
@click.option('--autotvm_early_stopping', default=100)
@click.option('--autotvm_log', default="mm_autotvm.log", type=click.Path())
@click.option('--tracker_port', default=9195)
@click.option('--device', type=click.Choice(TARGETS.keys()))
def tune(
        autotvm_number,
        autotvm_repeat,
        autotvm_n_trial,
        autotvm_early_stopping,
        autotvm_log,
        tracker_port,
        device):
    logging.basicConfig(level=logging.DEBUG)
    # logging.getLogger('autotvm').setLevel(logging.DEBUG)
    # logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))
    task = autotvm.task.create(
        mm_NCn,
        args=(M, N, K, P, dtype),
        target=TARGETS[device]
    )
    print(task.config_space)
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.RPCRunner(
            key="skl",
            host="localhost",
            port=9195,
            number=autotvm_number,
            repeat=autotvm_repeat
        ) if device == "skl" else autotvm.LocalRunner(
            number=autotvm_number,
            repeat=autotvm_repeat
        )
    )

    tuner = autotvm.tuner.XGBTuner(task, feature_type="knob")
    tuner.tune(
        n_trial=min(autotvm_n_trial, len(task.config_space)),
        early_stopping=autotvm_early_stopping,
        measure_option=measure_option,
        callbacks=[
            autotvm.callback.progress_bar(
                autotvm_n_trial,
                prefix="I"),
            autotvm.callback.log_to_file(autotvm_log)])



if __name__ == "__main__":
    tune()
