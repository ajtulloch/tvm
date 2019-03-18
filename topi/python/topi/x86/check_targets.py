# pylint: disable=invalid-name,unused-variable,invalid-name,unused-argument
"""Checks different x86 targets for target specific schedules"""

def check_skylake(target):
    """
    Checks if the target is skylake
    """

    for opt in target.options:
        if opt == '-mcpu=skylake-avx512':
            return True
    return False


def fp32_vector_width():
    """
    """
    import tvm
    target = tvm.target.current_target()
    if target is None:
        return 8

    for opt in target.options:
        if opt == '-mcpu=skylake-avx512':
            return 16
        if opt == '-mcpu=core-avx2':
            return 8
    return 8
