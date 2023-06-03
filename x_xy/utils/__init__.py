from .batchsize import distribute_batchsize, expand_batchsize, merge_batchsize
from .sys_composer import inject_system

JIT_WARN = True


def disable_jit_warn():
    global JIT_WARN
    JIT_WARN = False
