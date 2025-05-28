##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

import ctypes
import glob
import math
import os
import re
import shutil
import threading
from abc import abstractmethod
from pathlib import Path

import pandas as pd
import yaml

import config
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
)
from utils.mi_gpu_spec import mi_gpu_specs
from utils.parser import build_in_vars, supported_denom
from utils.utils import (
    capture_subprocess_output,
    convert_metric_id_to_panel_idx,
    detect_rocprof,
    get_base_spi_pipe_counter,
    get_submodules,
    is_spi_pipe_counter,
    is_tcc_channel_counter,
    using_v3,
)


class OmniSoC_Base:
    def __init__(
        self, args, mspec
    ):  # new info field will contain rocminfo or sysinfo to populate properties
        console_debug("[omnisoc init]")
        self.__args = args
        self.__arch = None
        self._mspec = mspec
        self.__perfmon_dir = None
        self.__perfmon_config = (
            {}
        )  # Per IP block max number of simulutaneous counters. GFX IP Blocks
        self.__soc_params = {}  # SoC specifications
        self.__compatible_profilers = []  # Store profilers compatible with SoC
        self.populate_mspec()

    def __hash__(self):
        return hash(self.__arch)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.__arch == other.get_soc()

    def set_perfmon_dir(self, path: str):
        self.__perfmon_dir = path

    def set_perfmon_config(self, config: dict):
        self.__perfmon_config = config

    def get_soc_param(self):
        return self.__soc_params

    def set_arch(self, arch: str):
        self.__arch = arch

    def get_arch(self):
        return self.__arch

    def get_args(self):
        return self.__args

    def set_compatible_profilers(self, profiler_names: list):
        self.__compatible_profilers = profiler_names

    def get_compatible_profilers(self):
        return self.__compatible_profilers

    def populate_mspec(self):
        from utils.specs import run, search, total_sqc

        if not hasattr(self._mspec, "_rocminfo") or self._mspec._rocminfo is None:
            return

        # load stats from rocminfo
        self._mspec.gpu_l1 = ""
        self._mspec.gpu_l2 = ""
        for idx2, linetext in enumerate(self._mspec._rocminfo):
            key = search(r"^\s*L1:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key != None:
                self._mspec.gpu_l1 = key
                continue

            key = search(r"^\s*L2:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key != None:
                self._mspec.gpu_l2 = key
                continue

            key = search(r"^\s*Max Clock Freq\. \(MHz\):\s+([0-9]+)", linetext)
            if key != None:
                self._mspec.max_sclk = key
                continue

            key = search(r"^\s*Compute Unit:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key != None:
                self._mspec.cu_per_gpu = key
                continue

            key = search(r"^\s*SIMDs per CU:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key != None:
                self._mspec.simd_per_cu = key
                continue

            key = search(r"^\s*Shader Engines:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key != None:
                self._mspec.se_per_gpu = key
                continue

            key = search(r"^\s*Wavefront Size:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key != None:
                self._mspec.wave_size = key
                continue

            key = search(r"^\s*Workgroup Max Size:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key != None:
                self._mspec.workgroup_max_size = key
                continue

            key = search(r"^\s*Max Waves Per CU:\s+ ([a-zA-Z0-9]+)\s*", linetext)
            if key != None:
                self._mspec.max_waves_per_cu = key
                break

        self._mspec.sqc_per_gpu = str(
            total_sqc(
                self._mspec.gpu_arch, self._mspec.cu_per_gpu, self._mspec.se_per_gpu
            )
        )

        # we get the max mclk from rocm-smi --showmclkrange
        # Regular expression to extract the max memory clock (third frequency level in MEM)
        memory_clock_pattern = (
            r"MEM:\s*[^:]*FREQUENCY_LEVELS:\s*(?:\d+: \d+ MHz\s*){2}(\d+)\s*MHz"
        )
        amd_smi_mclk = run(["amd-smi", "static"], exit_on_error=True)
        self._mspec.max_mclk = search(memory_clock_pattern, amd_smi_mclk)

        console_debug("max mem clock is {}".format(self._mspec.max_mclk))

        # these are just max's now, because the parsing was broken and this was inconsistent
        # with how we use the clocks elsewhere (all max, all the time)
        self._mspec.cur_sclk = self._mspec.max_sclk
        self._mspec.cur_mclk = self._mspec.max_mclk

        self._mspec.gpu_series = mi_gpu_specs.get_gpu_series(self._mspec.gpu_arch)
        # specify gpu model name for gfx942 hardware
        self._mspec.gpu_model = mi_gpu_specs.get_gpu_model(
            self._mspec.gpu_arch, self._mspec.gpu_chip_id
        )
        self._mspec.num_xcd = str(
            mi_gpu_specs.get_num_xcds(
                self._mspec.gpu_model, self._mspec.compute_partition
            )
        )

    @demarcate
    def detect_counters(self):
        """
        Create a set of counters required for the selected report sections.
        Parse analysis report configuration files based on the selected report sections to be filtered.
        """
        counters = set()
        config_filenames = {
            filename: []
            for filename in os.listdir(
                Path(self.get_args().config_dir).joinpath(self.__arch)
            )
            if filename.endswith(".yaml")
        }
        metric_ids = [
            name
            for name, type in self.get_args().filter_blocks.items()
            if type == "metric_id"
        ]
        file_ids = []
        for section in metric_ids:
            section_num = convert_metric_id_to_panel_idx(section)
            file_id = str(section_num // 100)
            # Convert "4" to "04"
            if len(file_id) == 1:
                file_id = f"0{file_id}"
            file_ids.append(file_id)
            # Apply sub section filtering
            for config_filename in config_filenames:
                if config_filename.startswith(file_id) and section_num % 100:
                    config_filenames[config_filename].append(section_num)

        # Apply section filters only if metric ids have been provided for filtering
        if metric_ids:
            # Identify yaml files corresponding to file_ids
            config_filenames = {
                filename: subsections
                for filename, subsections in config_filenames.items()
                if filename.startswith(tuple(file_ids))
            }

        for config_filename, subsections in config_filenames.items():
            # Read the yaml file
            with open(
                Path(self.get_args().config_dir).joinpath(self.__arch, config_filename),
                "r",
            ) as stream:
                section_config = yaml.safe_load(stream)
            # Extract subsection if section is of the form 4.52
            if subsections:
                section_config_text = "\n".join(
                    [
                        # Convert yaml to string
                        yaml.dump(subsection)
                        for subsection in section_config["Panel Config"]["data source"]
                        if subsection["metric_table"]["id"] in subsections
                    ]
                )
            else:
                # Convert yaml to string
                section_config_text = yaml.dump(section_config)
            counters = counters.union(self.parse_counters(section_config_text))

        # Handle TCC channel counters: if hw_counter_matches has elements ending with '['
        # Expand and interleve the TCC channel counters
        # e.g.  TCC_HIT[0] TCC_ATOMIC[0] ... TCC_HIT[1] TCC_ATOMIC[1] ...
        num_xcd_for_pmc_file = 1
        if using_v3():
            num_xcd_for_pmc_file = int(self._mspec.num_xcd)

        for counter_name in counters.copy():
            if counter_name.startswith("TCC") and counter_name.endswith("["):
                counters.remove(counter_name)
                counter_name = counter_name.split("[")[0]
                counters = counters.union(
                    {
                        f"{counter_name}[{i}]"
                        for i in range(num_xcd_for_pmc_file * int(self._mspec._l2_banks))
                    }
                )

        return counters

    @demarcate
    def perfmon_filter(self, roofline_perfmon_only: bool):
        """Filter default performance counter set based on user arguments"""
        if (
            roofline_perfmon_only
            and Path(self.get_args().path).joinpath("pmc_perf.csv").is_file()
        ):
            return

        if roofline_perfmon_only:
            counters = set()
            for fname in glob.glob(self.__perfmon_dir + "/" + "pmc_roof_perf.txt"):
                lines = open(fname, "r").read().splitlines()
                for line in lines:
                    # Strip all comments, skip empty lines
                    stext = line.split("#")[0].strip()
                    if not stext:
                        continue
                    # all pmc counters start with  "pmc:"
                    m = re.match(r"^pmc:(.*)", stext)
                    if m is None:
                        continue
                    # de-duplicate counters
                    counters = counters.union(set(m.group(1).split()))
        else:
            counters = self.detect_counters()
            # Perfmon hardware block filtering
            filter_hardware_blocks = [
                name
                for name, type in self.get_args().filter_blocks.items()
                if type == "hardware_block"
            ]
            if filter_hardware_blocks:
                counters = {
                    counter_name
                    for counter_name in counters
                    if counter_name.startswith(tuple(filter_hardware_blocks))
                }

        if not using_v3():
            # Counters not supported in rocprof v1 / v2
            counters = counters - {"SQ_INSTS_VALU_MFMA_F8", "SQ_INSTS_VALU_MFMA_MOPS_F8"}

        # Following counters are not supported
        # TCP_TCP_LATENCY_sum (except for gfx950)
        # SQC_DCACHE_INFLIGHT_LEVEL
        counters = counters - {"SQC_DCACHE_INFLIGHT_LEVEL"}
        if self.__arch != "gfx950":
            counters = counters - {"TCP_TCP_LATENCY_sum"}

        # SQ_ACCUM_PREV_HIRES will be injected for level counters later on
        counters = counters - {"SQ_ACCUM_PREV_HIRES"}

        # Coalesce and writeback workload specific perfmon
        self.perfmon_coalesce(counters)

    @demarcate
    def parse_counters(self, config_text):
        """
        Create a set of all hardware counters mentioned in the given config file content string
        """
        hw_counter_matches, variable_matches = self.parse_counters_text(config_text)

        # get hw counters and variables for all supported denominators
        for formula in supported_denom.values():
            hw_counter_matches_denom, variable_matches_denom = self.parse_counters_text(
                formula
            )
            hw_counter_matches.update(hw_counter_matches_denom)
            variable_matches.update(variable_matches_denom)

        # get hw counters corresponding to variables recursively
        while variable_matches:
            subvariable_matches = set()
            for var in variable_matches:
                if var in build_in_vars:
                    hw_counter_matches_vars, variable_matches_vars = (
                        self.parse_counters_text(build_in_vars[var])
                    )
                    hw_counter_matches.update(hw_counter_matches_vars)
                    subvariable_matches.update(variable_matches_vars)
            # process new found variables
            variable_matches = subvariable_matches - variable_matches

        return hw_counter_matches

    def parse_counters_text(self, text):
        """Parse out hardware counters and variables from given text"""
        # hw counter name should start with ip block name
        # hw counter name should have all capital letters or digits and should not end with underscore
        # he counter name can either optionally end with '[' or '_sum'
        hw_counter_regex = (
            r"(?:SQ|SQC|TA|TD|TCP|TCC|CPC|CPF|SPI|GRBM)_[0-9A-Z_]*[0-9A-Z](?:\[|_sum)*"
        )
        # only capture the variable name after $ using capturing group
        variable_regex = r"\$([0-9A-Za-z_]*[0-9A-Za-z])"
        hw_counter_matches = set(re.findall(hw_counter_regex, text))
        variable_matches = set(re.findall(variable_regex, text))
        # variable matches cannot be counters
        hw_counter_matches = hw_counter_matches - variable_matches
        return hw_counter_matches, variable_matches

    def get_rocprof_supported_counters(self):
        rocprof_cmd = detect_rocprof(self.get_args())
        rocprof_counters = set()

        if str(rocprof_cmd).endswith("rocprof"):
            command = [rocprof_cmd, "--list-basic"]
            success, output = capture_subprocess_output(command, enable_logging=False)
            # return code should be 1 so success should be False
            if success:
                console_error(
                    f"Failed to list rocprof supported counters using command: {command}"
                )
            for line in output.splitlines():
                if "gpu-agent" in line:
                    counters, _ = self.parse_counters_text(line.split(":")[1].strip())
                    rocprof_counters.update(counters)

            command = [rocprof_cmd, "--list-derived"]
            success, output = capture_subprocess_output(command, enable_logging=False)
            # return code should be 1 so success should be False
            if success:
                console_error(
                    f"Failed to list rocprof supported counters using command: {command}"
                )
            for line in output.splitlines():
                if "gpu-agent" in line:
                    counters, _ = self.parse_counters_text(line.split(":")[1].strip())
                    rocprof_counters.update(counters)

        elif str(rocprof_cmd).endswith("rocprofv2"):
            command = [rocprof_cmd, "--list-counters"]
            success, output = capture_subprocess_output(command, enable_logging=False)
            # return code should be 1 so success should be False
            if success:
                console_error(
                    f"Failed to list rocprof supported counters using command: {command}"
                )
            for line in output.splitlines():
                if "gfx" in line:
                    counters, _ = self.parse_counters_text(line.split(":")[2].strip())
                    rocprof_counters.update(counters)

        elif str(rocprof_cmd).endswith("rocprofv3"):
            command = [rocprof_cmd, "--list-avail"]
            success, output = capture_subprocess_output(command, enable_logging=False)
            # return code should be 0 so success should be True
            if not success:
                console_error(
                    f"Failed to list rocprof supported counters using command: {command}"
                )
            for line in output.splitlines():
                if "Name:" in line:
                    counters, _ = self.parse_counters_text(line.split(":")[1].strip())
                    rocprof_counters.update(counters)

        elif str(rocprof_cmd) == "rocprofiler-sdk":
            MAX_STR = 256

            # rocprofiler sdk list avail library
            libname = str(
                Path(self.get_args().rocprofiler_sdk_library_path).parent.parent.joinpath(
                    "libexec/rocprofiler-sdk/librocprofv3-list-avail.so"
                )
            )
            c_lib = ctypes.CDLL(libname)
            if c_lib is None:
                console_error(f"Error opening {libname}")

            # Intialize the library and set data types for arguments and variables
            c_lib.avail_tool_init()
            c_lib.get_number_of_agents.restype = ctypes.c_size_t
            c_lib.get_agent_node_id.restype = ctypes.c_ulong
            c_lib.get_agent_node_id.argtypes = [ctypes.c_int]
            c_lib.get_number_of_counters.restype = ctypes.c_ulong
            c_lib.get_number_of_counters.argtypes = [ctypes.c_int]
            c_lib.get_counters_info.argtypes = [
                ctypes.c_ulong,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_ulong),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_char * MAX_STR)),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_char * MAX_STR)),
                ctypes.POINTER(ctypes.c_int),
            ]
            c_lib.get_counter_block.argtypes = [
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_char * MAX_STR)),
            ]

            # Iterate through each counter index and get its information
            for idx in range(c_lib.get_number_of_agents()):
                node_id = c_lib.get_agent_node_id(idx)
                for counter_idx in range(c_lib.get_number_of_counters(node_id)):
                    # Counter information will be stored in these variables
                    name_args = ctypes.POINTER(ctypes.c_char * MAX_STR)()
                    description_args = ctypes.POINTER(ctypes.c_char * MAX_STR)()
                    is_derived_args = ctypes.c_int()
                    counter_id_args = ctypes.c_ulong()
                    block_args = ctypes.POINTER(ctypes.c_char * MAX_STR)()
                    # Get the counter information
                    c_lib.get_counters_info(
                        node_id,
                        counter_idx,
                        ctypes.byref(counter_id_args),
                        name_args,
                        description_args,
                        ctypes.byref(is_derived_args),
                    )
                    c_lib.get_counter_block(node_id, counter_idx, block_args)
                    block = ctypes.cast(block_args, ctypes.c_char_p).value.decode("utf-8")
                    if not is_derived_args.value and block:
                        # Only consider raw hardware counters from IP blocks
                        rocprof_counters.add(
                            ctypes.cast(name_args, ctypes.c_char_p).value.decode("utf-8")
                        )

        else:
            console_error(
                "Incompatible profiler: %s. Supported profilers include: %s"
                % (rocprof_cmd, get_submodules("rocprof_compute_profile"))
            )

        return rocprof_counters

    @demarcate
    def perfmon_coalesce(self, counters):
        """Sort and bucket all related performance counters to minimize required application passes"""

        # Create workload directory
        # In some cases (i.e. --specs) path will not be given
        if hasattr(self.get_args(), "path"):
            if self.get_args().path == str(Path(os.getcwd()).joinpath("workloads")):
                workload_dir = str(
                    Path(self.get_args().path).joinpath(
                        self.get_args().name, self._mspec.gpu_model
                    )
                )
            else:
                workload_dir = self.get_args().path

        # Initialize directories
        if not Path(workload_dir).is_dir():
            os.makedirs(workload_dir)
        elif not Path(workload_dir).is_symlink():
            shutil.rmtree(workload_dir)
        else:
            os.unlink(workload_dir)

        workload_perfmon_dir = workload_dir + "/perfmon"
        os.makedirs(workload_perfmon_dir)

        # Sanity check whether counters are supported by underlying rocprof tool
        rocprof_counters = self.get_rocprof_supported_counters()
        # rocprof does not support TCC channel counters, so remove channel suffix for comparison
        not_supported_counters = {
            counter.split("[")[0] if is_tcc_channel_counter(counter) else counter
            for counter in counters
        } - rocprof_counters
        if not_supported_counters:
            console_warning(
                f"Following counters might not be supported by rocprof: {', '.join(not_supported_counters)} "
            )
        # We might be providing definitions of unsupported counters, so still try to collect them
        if not counters:
            console_error(
                "profiling",
                "No performance counters to collect, please check the provided profiling filters",
            )
        else:
            console_log(f"Collecting following counters: {', '.join(counters)} ")

        output_files = []

        accu_file_count = 0
        # Create separate perfmon file for LEVEL counters without _sum suffix
        # TCC LEVEL counters are handled channel wise, so ignore them
        for counter in counters.copy():
            if (
                "LEVEL" in counter
                and not counter.endswith("_sum")
                and not is_tcc_channel_counter(counter)
            ):
                counters.remove(counter)
                output_files.append(CounterFile(counter + ".txt", self.__perfmon_config))
                output_files[-1].add(counter)
                if using_v3():
                    # v3 does not support SQ_ACCUM_PREV_HIRES. Instead we defined our own
                    # counters in counter_defs.yaml that use the accumulate() function. These
                    # use the name of the accumulate counter with _ACCUM appended to them.
                    output_files[-1].add(counter + "_ACCUM")
                else:
                    output_files[-1].add("SQ_ACCUM_PREV_HIRES")
                accu_file_count += 1

        file_count = 0
        # Store all channels for a TCC channel counter in the same file
        tcc_channel_counter_file_map = dict()
        # Store all pipes for SPI pipe counters in the same file
        spi_pipe_counter_file_map = dict()
        for ctr in counters:
            # Store all channels for a TCC channel counter in the same file
            if is_tcc_channel_counter(ctr):
                output_file = tcc_channel_counter_file_map.get(ctr.split("[")[0])
                if output_file:
                    output_file.add(ctr)
                    continue
            # Store all pipes for SPI pipe counters in the same file
            if is_spi_pipe_counter(ctr):
                output_file = spi_pipe_counter_file_map.get(
                    get_base_spi_pipe_counter(ctr)
                )
                if output_file:
                    output_file.add(ctr)
                    continue
            # Add counter to first file that has room
            added = False
            for i in range(len(output_files)):
                if output_files[i].add(ctr):
                    added = True
                    # Store all channels for a TCC channel counter in the same file
                    if is_tcc_channel_counter(ctr):
                        tcc_channel_counter_file_map[ctr.split("[")[0]] = output_files[i]
                    # Store all pipes for SPI pipe counters in the same file
                    if is_spi_pipe_counter(ctr):
                        spi_pipe_counter_file_map[get_base_spi_pipe_counter(ctr)] = (
                            output_files[i]
                        )
                    break

            # All files are full, create a new file
            if not added:
                output_files.append(
                    CounterFile(
                        "pmc_perf_{}.txt".format(file_count), self.__perfmon_config
                    )
                )
                file_count += 1
                output_files[-1].add(ctr)

        console_debug("profiling", "perfmon_coalesce file_count %s" % file_count)

        # TODO: rewrite the above logic for spatial_multiplexing later
        if self.get_args().spatial_multiplexing:

            # TODO: more error checking
            if len(self.get_args().spatial_multiplexing) != 3:
                console_error(
                    "profiling",
                    "multiplexing need provide node_idx node_count and gpu_count",
                )

            node_idx = int(self.get_args().spatial_multiplexing[0])
            node_count = int(self.get_args().spatial_multiplexing[1])
            gpu_count = int(self.get_args().spatial_multiplexing[2])

            old_group_num = file_count + accu_file_count
            new_bucket_count = node_count * gpu_count
            groups_per_bucket = math.ceil(
                old_group_num / new_bucket_count
            )  # It equals to file num per node
            max_groups_per_node = groups_per_bucket * gpu_count

            group_start = node_idx * max_groups_per_node
            group_end = min((node_idx + 1) * max_groups_per_node, old_group_num)

            console_debug(
                "profiling",
                "spatial_multiplexing node_idx %s, node_count %s, gpu_count: %s, old_group_num %s, "
                "new_bucket_count %s, groups_per_bucket %s, max_groups_per_node %s, "
                "group_start %s, group_end %s"
                % (
                    node_idx,
                    node_count,
                    gpu_count,
                    old_group_num,
                    new_bucket_count,
                    groups_per_bucket,
                    max_groups_per_node,
                    group_start,
                    group_end,
                ),
            )

            for f_idx in range(groups_per_bucket):
                file_name = str(
                    Path(workload_perfmon_dir).joinpath(
                        "pmc_perf_" + "node_" + str(node_idx) + "_" + str(f_idx) + ".txt"
                    )
                )

                pmc = []
                for g_idx in range(
                    group_start + f_idx * gpu_count,
                    min(group_end, group_start + (f_idx + 1) * gpu_count),
                ):
                    gpu_idx = g_idx % gpu_count
                    for block_name in output_files[g_idx].blocks.keys():
                        for ctr in output_files[g_idx].blocks[block_name].elements:
                            pmc.append(ctr + ":device=" + str(gpu_idx))

                stext = "pmc: " + " ".join(pmc)

                # Write counters to file
                fd = open(file_name, "w")
                fd.write(stext + "\n\n")
                fd.close()

        else:
            # Output to files
            with open(
                str(
                    Path(config.rocprof_compute_home).joinpath(
                        "rocprof_compute_soc",
                        "profile_configs",
                        "accum_counters.yaml",
                    )
                ),
                "r",
            ) as fp:
                accum_counters_def = yaml.safe_load(fp)

            for f in output_files:
                file_name_txt = str(Path(workload_perfmon_dir).joinpath(f.file_name_txt))
                file_name_yaml = str(
                    Path(workload_perfmon_dir).joinpath(f.file_name_yaml)
                )

                pmc = []
                counter_def = dict()
                for ctr in [
                    ctr
                    for block_name in f.blocks
                    for ctr in f.blocks[block_name].elements
                ]:
                    pmc.append(ctr)
                    if using_v3():
                        if ctr in accum_counters_def:
                            counter_def[ctr] = accum_counters_def[ctr]
                        # Add TCC channel counters definitions
                        if is_tcc_channel_counter(ctr):
                            counter_name = ctr.split("[")[0]
                            idx = int(ctr.split("[")[1].split("]")[0])
                            xcd_idx = idx // int(self._mspec._l2_banks)
                            channel_idx = idx % int(self._mspec._l2_banks)
                            counter_def.update(
                                {
                                    ctr: {
                                        "architectures": {
                                            self.__arch: {
                                                "expression": f"select({counter_name},[DIMENSION_XCC=[{xcd_idx}], DIMENSION_INSTANCE=[{channel_idx}]])",
                                            }
                                        },
                                        "description": f"{counter_name} on {xcd_idx}th XCC and {channel_idx}th channel",
                                    }
                                }
                            )

                stext = "pmc: " + " ".join(pmc)
                # Write counters to file
                fd = open(file_name_txt, "w")
                fd.write(stext + "\n\n")
                fd.write("gpu:\n")
                fd.write("range:\n")
                fd.write("kernel:\n")
                fd.close()

                # Write counter definitions to file
                if using_v3():
                    with open(file_name_yaml, "w") as fp:
                        if counter_def:
                            fp.write(yaml.dump(counter_def))

        # Add a timestamp file
        # TODO: Does v3 need this?
        if not using_v3():
            fd = open(str(Path(workload_perfmon_dir).joinpath("timestamps.txt")), "w")
            fd.write("pmc:\n\n")
            fd.write("gpu:\n")
            fd.write("range:\n")
            fd.write("kernel:\n")
            fd.close()

    # ----------------------------------------------------
    # Required methods to be implemented by child classes
    # ----------------------------------------------------
    @abstractmethod
    def profiling_setup(self):
        """Perform any SoC-specific setup prior to profiling."""
        console_debug("profiling", "perform SoC profiling setup for %s" % self.__arch)

    @abstractmethod
    def post_profiling(self):
        """Perform any SoC-specific post profiling activities."""
        console_debug("profiling", "perform SoC post processing for %s" % self.__arch)

    @abstractmethod
    def analysis_setup(self):
        """Perform any SoC-specific setup prior to analysis."""
        console_debug("analysis", "perform SoC analysis setup for %s" % self.__arch)


# Set with limited size
class LimitedSet:
    def __init__(self, maxsize) -> None:
        self.avail = maxsize
        self.elements = []

    def add(self, e) -> None:
        if e in self.elements:
            return True
        # Store all channels for a TCC channel counter in the same file
        if e.split("[")[0] in {element.split("[")[0] for element in self.elements}:
            self.elements.append(e)
            return True
        # Store all pipes for SPI pipe counters in the same file
        if is_spi_pipe_counter(e) and get_base_spi_pipe_counter(e) in {
            get_base_spi_pipe_counter(element) for element in self.elements
        }:
            self.elements.append(e)
            return True
        if self.avail > 0:
            # SPI pipe counters take space of 2 counters
            if is_spi_pipe_counter(e):
                self.avail -= 2
            else:
                self.avail -= 1
            self.elements.append(e)
            return True
        return False


# Represents a file that lists PMC counters. Number of counters for each
# block limited according to perfmon config.
class CounterFile:
    def __init__(self, name, perfmon_config) -> None:
        name_no_extension = name.split(".")[0]
        self.file_name_txt = name_no_extension + ".txt"
        self.file_name_yaml = name_no_extension + ".yaml"
        self.blocks = {b: LimitedSet(v) for b, v in perfmon_config.items()}

    def add(self, counter) -> bool:
        block = counter.split("_")[0]

        # SQ and SQC belong to the same IP block
        if block == "SQC":
            block = "SQ"

        return self.blocks[block].add(counter)
