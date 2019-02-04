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
def bmm_packed(PB, N, K, M, P, dtype):
    A = tvm.placeholder((PB, N, K, P), name='A', dtype=dtype)
    B = tvm.placeholder((PB, K, M, P), name='B', dtype=dtype)
    k = tvm.reduce_axis((0, K), name='k')
    C = tvm.compute(
        (PB, N, M, P),
        lambda b, x, y, p: tvm.sum(A[b, x, k, p] * B[b, k, y, p], axis=k),
        tag='batch_matmul_packed'
    )

    s = tvm.create_schedule(C.op)
    ab, ax, ay, ap = s[C].op.axis
    ak, = s[C].op.reduce_axis

    cfg = autotvm.get_config()
    axo, axi = cfg.define_split('tile_x', ax, num_outputs=2)
    ayo, ayi = cfg.define_split('tile_y', ay, num_outputs=2)
    ako, aki = cfg.define_split('tile_k', ak, num_outputs=2)

    axo_, axi_, ayo_, ayi_, ako_, aki_ = axo, axi, ayo, ayi, ako, aki
    ab_, ap_ = cfg.axis(ab), cfg.axis(ap)

    cfg.define_reorder(
        "reorder_0",
        [ab_, axo_, axi_, ayo_, ayi_, ako_, aki_, ap_],
        policy='candidate',
        candidate=[
            [ab_, axo_, ayo_, ako_, aki_, axi_, ayi_, ap_],
            [ab_, axo_, ayo_, ako_, axi_, aki_, ayi_, ap_],
            [ab_, axo_, ayo_, ako_, axi_, ayi_, aki_, ap_],
            [ab_, axo_, ayo_, ako_, aki_, axi_, ayi_, ap_],
        ]
    )

    cfg.define_annotate("ann_spatial", [axi, ayi], policy='try_unroll')
    axo, axi = cfg['tile_x'].apply(s, C, ax)
    ayo, ayi = cfg['tile_y'].apply(s, C, ay)
    ako, aki = cfg['tile_k'].apply(s, C, ak)
    cfg['reorder_0'].apply(s, C, [ab, axo, axi, ayo, ayi, ako, aki, ap])
    cfg['ann_spatial'].apply(
        s,
        C,
        [axi, ayi],
        axis_lens=[
            cfg['tile_x'].size[-1],
            cfg['tile_x'].size[-1]
        ],
        max_unroll=16,
        cfg=cfg,
    )
    s[C].vectorize(ap)
    return s, [A, B, C]

TARGETS = dict(
    skl='llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu',
    local='llvm -mcpu=core-avx2'
)

dtype = "float32"

PB = 1
N = 40
M = 40
K = 160
P = 16

@click.command()
@click.option('--autotvm_number', default=50)
@click.option('--autotvm_repeat', default=4)
@click.option('--autotvm_n_trial', default=200)
@click.option('--autotvm_early_stopping', default=100)
@click.option('--autotvm_log', default="bmm_autotvm.log", type=click.Path())
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
        bmm_packed,
        args=(PB, N, K, M, P, dtype),
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
