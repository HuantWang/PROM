import os
import numpy as np

import tvm
from tvm import relay, autotvm
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_runtime as runtime

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif "vgg" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.vgg.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "squeezenet_v1.1":
        mod, params = relay.testing.squeezenet.get_workload(
            batch_size=batch_size, version="1.1", dtype=dtype
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={input_name: input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape


# Replace "llvm" with the correct target of your CPU.
# For example, for AWS EC2 c5 instance with Intel Xeon
# Platinum 8000 series, the target should be "llvm -mcpu=skylake-avx512".
# For AWS EC2 c4 instance with Intel Xeon E5-2666 v3, it should be
# "llvm -mcpu=core-avx2".
target = "llvm"

batch_size = 1
dtype = "float32"
model_name = "resnet-18"
log_file = "%s.log" % model_name
graph_opt_sch_file = "%s_graph_opt.log" % model_name

# Set the input name of the graph
# For ONNX models, it is typically "0".
input_name = "data"

# Set number of threads used for tuning based on the number of
# physical CPU cores on your machine.
num_threads = 1
os.environ["TVM_NUM_THREADS"] = str(num_threads)

tuning_option = {
    "log_filename": log_file,
    "tuner": "random",
    "early_stopping": None,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(
            number=1, repeat=1, min_repeat_ms=0, enable_cpu_cache_flush=True
        ),
    ),
}


# You can skip the implementation of this function for this tutorial.
def tune_kernels(
    tasks, measure_option, tuner="gridsearch", early_stopping=None, log_filename="tuning.log"
):

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb":
            tuner_obj = XGBTuner(task, loss_type="reg")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(task, loss_type="reg", feature_type="knob")
        elif tuner == "xgb_itervar":
            tuner_obj = XGBTuner(task, loss_type="reg", feature_type="itervar")
        elif tuner == "xgb_curve":
            tuner_obj = XGBTuner(task, loss_type="reg", feature_type="curve")
        elif tuner == "xgb_rank":
            tuner_obj = XGBTuner(task, loss_type="rank")
        elif tuner == "xgb_rank_knob":
            tuner_obj = XGBTuner(task, loss_type="rank", feature_type="knob")
        elif tuner == "xgb_rank_itervar":
            tuner_obj = XGBTuner(task, loss_type="rank", feature_type="itervar")
        elif tuner == "xgb_rank_curve":
            tuner_obj = XGBTuner(task, loss_type="rank", feature_type="curve")
        elif tuner == "xgb_rank_binary":
            tuner_obj = XGBTuner(task, loss_type="rank-binary")
        elif tuner == "xgb_rank_binary_knob":
            tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="knob")
        elif tuner == "xgb_rank_binary_itervar":
            tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="itervar")
        elif tuner == "xgb_rank_binary_curve":
            tuner_obj = XGBTuner(task, loss_type="rank-binary", feature_type="curve")
        elif tuner == "ga":
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(task)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial = len(task.config_space)
        tuner_obj.tune(
            n_trial=n_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_filename),
            ],
        )


# Use graph tuner to achieve graph level optimal schedules
# Set use_DP=False if it takes too long to finish.
def tune_graph(graph, dshape, records, opt_sch_file, use_DP=True):
    target_op = [
        relay.op.get("nn.conv2d"),
    ]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {input_name: dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)

def evaluate_performance(lib, data_shape):
    # upload parameters to device
    dev = tvm.cpu()
    data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
    module = runtime.GraphModule(lib["default"](dev))
    module.set_input(input_name, data_tvm)

    # evaluate
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, number=100, repeat=3))


def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, data_shape, out_shape = get_network(model_name, batch_size)
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )

    # run tuning tasks
    tune_kernels(tasks, **tuning_opt)
    tune_graph(mod["main"], data_shape, log_file, graph_opt_sch_file)

    # compile kernels in default mode
    print("Evaluation of the network compiled in 'default' mode without auto tune:")
    with tvm.transform.PassContext(opt_level=3):
        print("Compile...")
        lib = relay.build(mod, target=target, params=params)
        evaluate_performance(lib, data_shape)

    # compile kernels in kernel tuned only mode
    print("\nEvaluation of the network been tuned on kernel level:")
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
        evaluate_performance(lib, data_shape)

    # compile kernels with graph-level best records
    print("\nEvaluation of the network been tuned on graph level:")
    with autotvm.apply_graph_best(graph_opt_sch_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)
        evaluate_performance(lib, data_shape)


# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

tune_and_evaluate(tuning_option)