# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Testcode for Android RPC.

To use it, start an RPC tracker with "python -m tvm.exec.rpc_tracker".
Use the tracker's address and port when configuring the RPC app.
Use "android" as the key if you wish to avoid modifying this script.
"""

import tvm
import os
from tvm import rpc
from tvm.contrib import util, ndk
import numpy as np

# override metal compiler to compile to iphone
@tvm.register_func("tvm_callback_vulkan_postproc")
def compile_metal(shader):
    with open("asdf.spv", "wb") as f:
        f.write(shader)
    return shader

# Set to be address of tvm proxy.
tracker_host = os.environ["TVM_TRACKER_HOST"]
tracker_port = int(os.environ["TVM_TRACKER_PORT"])
key = "android"

# Change target configuration.
# Run `adb shell cat /proc/cpuinfo` to find the arch.
arch = "arm64"
target = "llvm -target=%s-linux-android" % arch

# whether enable to execute test on OpenCL target
test_opencl = False
# whether enable to execute test on Vulkan target
test_vulkan = True

def test_rpc_module():
    # graph
    n = 256 * 3 * 4 * 64 * 20 * 2
    dtype = 'float32'
    swish = False
    A = tvm.placeholder((n,), name='A', dtype=dtype)
    if not swish:
        B = tvm.compute(
            A.shape,
            lambda *i: A(*i) + tvm.expr.FloatImm(dtype, 1.0),
            name='B')
    else:
        B = tvm.compute(A.shape, lambda *i: A(*i) * tvm.max(
            tvm.min(
                A(*i) + tvm.expr.FloatImm(dtype, 3.0),
                tvm.expr.FloatImm(dtype, 6.0)),
            tvm.expr.FloatImm(dtype, 0.0)), name='B')
    a_np = np.random.uniform(size=n).astype(A.dtype)
    temp = util.tempdir()

    # Establish remote connection with target hardware
    print("Got tracker")
    tracker = rpc.connect_tracker(tracker_host, tracker_port)
    print("Requesting remote")
    print(key)
    remote = tracker.request(key, priority=1,
                             session_timeout=6000)
    print("Got remote")
    # # Compile the Graph for CPU target
    # s = tvm.create_schedule(B.op)
    # xo, xi = s[B].split(B.op.axis[0], factor=64)
    # s[B].parallel(xi)
    # s[B].pragma(xo, "parallel_launch_point")
    # s[B].pragma(xi, "parallel_barrier_when_finish")
    # f = tvm.build(s, [A, B], target, name="myadd_cpu")
    # path_dso_cpu = temp.relpath("cpu_lib.so")
    # f.export_library(path_dso_cpu, ndk.create_shared)

    # # Execute the portable graph on cpu target
    # print('Run CPU test ...')
    # ctx = remote.cpu(0)
    # remote.upload(path_dso_cpu)
    # f2 = remote.load_module("cpu_lib.so")
    # a = tvm.nd.array(a_np, ctx)
    # b = tvm.nd.array(np.zeros(n, dtype=A.dtype), ctx)
    # time_f = f2.time_evaluator(f2.entry_name, ctx, number=10)
    # cost = time_f(a, b).mean
    # print('%g secs/op\n' % cost)
    # np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

    # Compile the Graph for OpenCL target
    if test_opencl:
        s = tvm.create_schedule(B.op)
        xo, xi = s[B].split(B.op.axis[0], factor=64)
        s[B].bind(xi, tvm.thread_axis("threadIdx.x"))
        s[B].bind(xo, tvm.thread_axis("blockIdx.x"))
        # Build the dynamic lib.
        # If we don't want to do metal and only use cpu, just set target to be target
        f = tvm.build(s, [A, B], "opencl", target_host=target, name="myadd")
        path_dso_cl = temp.relpath("dev_lib_cl.so")
        f.export_library(path_dso_cl, ndk.create_shared)

        print('Run GPU(OpenCL Flavor) test ...')
        ctx = remote.cl(0)
        remote.upload(path_dso_cl)
        f1 = remote.load_module("dev_lib_cl.so")
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(n, dtype=A.dtype), ctx)
        time_f = f1.time_evaluator(f1.entry_name, ctx, min_repeat_ms=1000, repeat=5)
        cost = time_f(a, b).mean
        print('%g secs/op\n' % cost)
        # np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

    # Compile the Graph for Vulkan target
    if test_vulkan:
        for n_threads in [256, 512, 1024]:
            for vectorize in [1, 2, 3, 4]:
                s = tvm.create_schedule(B.op)
                xo, xi = s[B].split(B.op.axis[0], factor=n_threads * (4 * vectorize if vectorize else 1))

                s[B].bind(xo, tvm.thread_axis("blockIdx.x"))
                if vectorize:
                    xi, xii = s[B].split(xi, factor=4 * vectorize)
                    xiiu, xii = s[B].split(xii, factor=4)
                    s[B].unroll(xiiu)
                    s[B].vectorize(xii)
                s[B].bind(xi, tvm.thread_axis("threadIdx.x"))
                # Build the dynamic lib.
                # If we don't want to do metal and only use cpu, just set target to be target
                print(tvm.lower(s, [A, B], simple_mode=True))
                f = tvm.build(s, [A, B], "vulkan", target_host=target, name="myadd")
                fname = f"dev_lib_vulkan_{np.random.random()}.so"
                path_dso_vulkan = temp.relpath(fname)
                f.export_library(path_dso_vulkan, ndk.create_shared)
                print('Run GPU(Vulkan Flavor) test ...')
                print(f"Vectorize: {vectorize}, {n_threads}")
                ctx = remote.vulkan(0)
                remote.upload(path_dso_vulkan)
                f1 = remote.load_module(fname)
                a = tvm.nd.array(a_np, ctx)
                b = tvm.nd.array(np.zeros(n, dtype=A.dtype), ctx)
                time_f = f1.time_evaluator(f1.entry_name, ctx, min_repeat_ms=1000, repeat=5)
                cost = time_f(a, b).mean
                if swish:
                    def hswish(x):
                        return np.clip(x+3, 0, 6) * x
                    print((b.asnumpy()[:5], (hswish(a.asnumpy())[:5])))
                    try:
                        np.testing.assert_allclose(b.asnumpy(), hswish(a.asnumpy()))
                    except Exception as e:
                        print(e)
                print(f"{n * 2 * (4 if dtype == 'float32' else 2) / cost / 1.0e9} GB/s, {n * (4 if swish else 1) / cost / 1.0e9} GFLOP/s, {cost * 1.0e6:.2f} us")
                # print('%g secs/op\n' % cost)


if __name__ == "__main__":
    test_rpc_module()
