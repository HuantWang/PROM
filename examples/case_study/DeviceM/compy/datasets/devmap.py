import os
import pandas as pd
from tqdm import tqdm

from compy.datasets import dataset


class OpenCLDevmapDataset(dataset.Dataset):
    def __init__(self):
        super().__init__()

        uri = "http://wwwpub.zih.tu-dresden.de/~s9602232/devmap.zip"
        self.download_http_and_extract(uri)

        self.additional_include_dirs = [
            os.path.join(self.content_dir, "support/libclc")
        ]

    def preprocess(self, builder, visitor, suite, benchmark_suites=None):
        suite_specifics = suite
        if benchmark_suites is None:
            benchmark_suites = suite_specifics.keys()

        opencl_header = str.encode(
            '#include "' + self.content_dir + '/support/opencl-shim.h"\n'
        )
        basedir = os.path.join(self.content_dir, "src")

        # Load cgo17 dataset.
        df = pd.read_csv(os.path.join(self.content_dir, "data", "cgo17-amd.csv"))

        # Remove preprocessing data from it.
        for column in [
            "Unnamed: 0",
            "comp",
            "rational",
            "mem",
            "localmem",
            "coalesced",
            "atomic",
            "seq",
            "src",
        ]:
            del df[column]

        # Split benchmark name for better identification.
        for idx, row in df.iterrows():
            b = row["benchmark"]

            df.loc[idx, "function_name"] = b.split("-")[-1]
            df.loc[idx, "benchmark_name"] = b.split("-")[-2]
            df.loc[idx, "suite_name"] = "-".join(b.split("-")[0:-2])

        del df["benchmark"]
        df = df[df["suite_name"].isin(benchmark_suites)]

        # Build a map of files to process: {file_name: further infos e.g. aux inputs, mappings}
        to_process = {}
        for idx, row in df.iterrows():
            # if idx ==40:
            #     break
            if row["benchmark_name"] == 'leukocyte':
                    continue
            suite_data = suite_specifics[row["suite_name"]]

            # Build benchmark name
            benchmark_name = row["benchmark_name"]
            if "remappings" in suite_data:
                benchmark_name = suite_data["remappings"][row["benchmark_name"]]

            if "benchmark_name_prefix" in suite_data:
                benchmark_name = suite_data["benchmark_name_prefix"] + benchmark_name

            if row["suite_name"] == "shoc-1.1.5":
                benchmark_name = benchmark_name.lower()

            # Build subdir
            subdir = suite_data["subdir"]
            if row["suite_name"] == "shoc-1.1.5":
                if benchmark_name == "s3d":
                    subdir = os.path.join(subdir, "..", "level2")

            # Search for CL file
            # 1. Build search path
            bench_dir = os.path.join(basedir, row["suite_name"], subdir, benchmark_name)
            if row["suite_name"] == "parboil-0.2":
                bench_dir = os.path.join(bench_dir, "src", "opencl_base")
            assert os.path.isdir(bench_dir)

            # 2. Search.
            cls = []
            for folder, subfolders, files in os.walk(bench_dir):
                for file in files:
                    if file.endswith(".cl"):
                        cls.append(os.path.join(os.path.abspath(folder), file))
            assert len(cls) >= 1
            if len(cls) > 1:
                if "filename_matcher" in suite_data:
                    for cls_it in cls:
                        if suite_data["filename_matcher"] in os.path.basename(cls_it):
                            cls = [cls_it]
                            break

            for bench_file in cls:
                assert os.path.isfile(bench_file)

                df.loc[idx, "bench_file"] = bench_file

                with open(bench_file, "rb") as file:
                    source_code = file.read()

                # Additional data
                # - Include dirs
                additional_include_dir = bench_dir

                # Add to to_process.
                file_data = (
                    bench_file,
                    source_code,
                    additional_include_dir,
                    row["suite_name"],
                    row["benchmark_name"],
                )
                if file_data not in to_process:
                    to_process[file_data] = []

                function_data = (
                    row["function_name"],
                    row["transfer"],
                    row["wgsize"],
                    row["oracle"],
                    row["runtime_cpu"],
                    row["runtime_gpu"],
                )
                to_process[file_data].append(function_data)

        # Process the map of files
        processed = {}
        for file_data in to_process:
            (
                bench_file,
                source_code,
                additional_include_dir,
                suite_name,
                benchmark_name,
            ) = file_data

            # if benchmark_name=='leukocyte':
            #     continue
            # print("name : ",benchmark_name)
            extractionInfo = builder.string_to_info(
                opencl_header + source_code, additional_include_dir
            )

            for functionInfo in extractionInfo.functionInfos:
                # (_, _, _, _, runtime_cpu, runtime_gpu
                #  ) = function_data
                processed[
                    (suite_name, benchmark_name, functionInfo.name)
                ] = functionInfo

        # Map to dataset and extract representations
        samples = {}
        samples_test = {}
        for file_data, function_datas in to_process.items():
            (
                bench_file,
                source_code,
                additional_include_dir,
                suite_name,
                benchmark_name,
            ) = file_data

            for function_data in function_datas:
                function_name, transfer, wgsize, label, cpu_time, gpu_time = function_data

                item_info = (suite_name, benchmark_name, function_name)

                samples[
                    item_info + (transfer, wgsize, label, cpu_time, gpu_time)
                ] = builder.info_to_representation(processed[item_info], visitor)

        # print("Size of dataset:", len(samples))
        # print("Number of unique tokens:", builder.num_tokens())
        # builder.print_tokens()

        return {
            "samples": [
                {
                    "info": info,
                    "x": {"code_rep": sample, "aux_in": [info[3], info[4]]},
                    "y": 0 if info[5] == "CPU" else 1,
                    "cpu_time": info[6],
                    "gpu_time": info[7],
                }
                for info, sample in samples.items()
            ],
            "num_types": builder.num_tokens(),
        }
