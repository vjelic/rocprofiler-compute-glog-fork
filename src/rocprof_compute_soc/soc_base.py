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

import glob
import math
import os
import re
import shutil
import sys
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from utils.mi_gpu_spec import get_gpu_model, get_gpu_series
from utils.parser import build_in_vars, supported_denom
from utils.utils import (
    console_debug,
    console_error,
    console_log,
    convert_metric_id_to_panel_idx,
    demarcate,
    get_default_accumulate_counter_file_ymal,
    total_xcds,
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
        self.__section_counters = set()  # hw counters corresponding to filtered sections
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

    def get_workload_perfmon_dir(self):
        return str(Path(self.__perfmon_dir).parent.absolute())

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

    @demarcate
    def populate_mspec(self):
        console_debug("[populate_mspec]")
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

        self._mspec.gpu_series = get_gpu_series(self._mspec.gpu_arch).upper()
        # specify gpu model name for gfx942 hardware
        self._mspec.gpu_model = get_gpu_model(
            self._mspec.gpu_arch, self._mspec.gpu_chip_id
        ).upper()
        self._mspec.num_xcd = str(
            total_xcds(self._mspec.gpu_model, self._mspec.compute_partition)
        )

    @demarcate
    def section_filter(self):
        """
        Create a set of counters required for the selected report sections.
        Parse analysis report configuration files based on the selected report sections to be filtered.
        """
        args = self.__args
        for section in self.__filter_metric_ids:
            section_num = convert_metric_id_to_panel_idx(section)
            file_id = str(section_num // 100)
            # Convert "4" to "04"
            if len(file_id) == 1:
                file_id = f"0{file_id}"
            # Identify yaml file corresponding to file_id
            config_filename = [
                filename
                for filename in os.listdir(Path(args.config_dir).joinpath(self.__arch))
                if filename.endswith(".yaml") and filename.startswith(file_id)
            ][0]
            # Read the yaml file
            with open(
                Path(args.config_dir).joinpath(self.__arch, config_filename), "r"
            ) as stream:
                section_config = yaml.safe_load(stream)
            # Extract subsection if section is of the form 4.52
            if section_num % 100:
                section_config_text = "\n".join(
                    [
                        # Convert yaml to string
                        yaml.dump(subsection)
                        for subsection in section_config["Panel Config"]["data source"]
                        if subsection["metric_table"]["id"] == section_num
                    ]
                )
            else:
                # Convert yaml to string
                section_config_text = yaml.dump(section_config)
            self.__section_counters = self.__section_counters.union(
                parse_counters(section_config_text)
            )

    @demarcate
    def perfmon_filter(self, roofline_perfmon_only: bool):
        """Filter default performance counter set based on user arguments"""
        if (
            roofline_perfmon_only
            and Path(self.get_args().path).joinpath("pmc_perf.csv").is_file()
        ):
            return

        # In some cases (i.e. --specs) path will not be given
        if hasattr(self.__args, "path"):
            if self.__args.path == str(Path(os.getcwd()).joinpath("workloads")):
                workload_dir = str(
                    Path(self.__args.path).joinpath(
                        self.__args.name, self._mspec.gpu_model
                    )
                )
            else:
                workload_dir = self.__args.path

        workload_perfmon_dir = workload_dir + "/perfmon"

        self.__filter_hardware_blocks = [
            name
            for name, type in self.get_args().filter_blocks.items()
            if type == "hardware_block"
        ]
        self.__filter_metric_ids = [
            name
            for name, type in self.get_args().filter_blocks.items()
            if type == "metric_id"
        ]

        self.section_filter()

        # Initialize directories
        if not Path(workload_dir).is_dir():
            os.makedirs(workload_dir)
        elif not Path(workload_dir).is_symlink():
            shutil.rmtree(workload_dir)
        else:
            os.unlink(workload_dir)

        os.makedirs(workload_perfmon_dir)

        if not roofline_perfmon_only:
            ref_pmc_files_list = glob.glob(self.__perfmon_dir + "/" + "pmc_*perf*.txt")
            ref_pmc_files_list += glob.glob(
                self.__perfmon_dir + "/" + self.__arch + "/pmc_*_perf*.txt"
            )

            # Perfmon list filtering
            if self.__filter_hardware_blocks:
                hardware_blocks = [
                    block.lower() for block in self.__filter_hardware_blocks
                ]
                mpattern = "pmc_([a-zA-Z0-9_]+)_perf*"

                pmc_files_list = []
                for fname in ref_pmc_files_list:
                    fbase = Path(fname).stem
                    ip = re.match(mpattern, fbase).group(1)
                    if ip in hardware_blocks:
                        pmc_files_list.append(fname)
                        console_log("fname: " + fbase + ": Added")
                    else:
                        console_log("fname: " + fbase + ": Skipped")

            else:
                # default: take all perfmons
                pmc_files_list = ref_pmc_files_list
        else:
            ref_pmc_files_list = glob.glob(self.__perfmon_dir + "/" + "pmc_roof_perf.txt")
            pmc_files_list = ref_pmc_files_list

        # Coalesce and writeback workload specific perfmon
        perfmon_coalesce(
            pmc_files_list,
            self.__perfmon_config,
            workload_dir,
            self.get_args().spatial_multiplexing,
            self.__section_counters,
            self._mspec,
            self.__arch,
        )

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


def getblock(counter):
    return counter.split("_")[0]


# Set with limited size
class LimitedSet:
    def __init__(self, maxsize) -> None:
        self.avail = maxsize
        self.elements = []

    def add(self, e) -> None:
        if e in self.elements:
            return True
        elif self.avail <= 0:
            return False

        self.avail -= 1
        self.elements.append(e)

        return True


# Represents a file that lists PMC counters. Number of counters for each
# block limited according to perfmon config.
class CounterFile:
    def __init__(self, name, perfmon_config) -> None:
        name_no_extension = name.split(".")[0]
        self.file_name_txt = name_no_extension + ".txt"
        self.file_name_yaml = name_no_extension + ".yaml"
        self.blocks = {b: LimitedSet(v) for b, v in perfmon_config.items()}

    def add(self, counter) -> bool:
        block = getblock(counter)

        # SQ and SQC belong to the same IP block
        if block == "SQC":
            block = "SQ"

        return self.blocks[block].add(counter)


# FIXME: This is a HACK
def using_v3():
    return "ROCPROF" in os.environ.keys() and os.environ["ROCPROF"].endswith("rocprofv3")


@demarcate
def parse_counters(config_text):
    """
    Create a set of all hardware counters mentioned in the given config file content string
    """
    # hw counter name should start with ip block name
    hw_counter_regex = r"(?:SQ|SQC|TA|TD|TCP|TCC|CPC|CPF|SPI|GRBM)_[0-9A-Za-z_]+"
    # only capture the variable name after $ using capturing group
    variable_regex = r"\$([0-9A-Za-z_]+)"
    hw_counter_matches = set(re.findall(hw_counter_regex, config_text))
    variable_matches = set(re.findall(variable_regex, config_text))
    # get hw counters and variables for all supported denominators
    for formula in supported_denom.values():
        hw_counter_matches.update(re.findall(hw_counter_regex, formula))
        variable_matches.update(re.findall(variable_regex, formula))
    # get hw counters corresponding to variables recursively
    while variable_matches:
        subvariable_matches = set()
        for var in variable_matches:
            if var in build_in_vars:
                hw_counter_matches.update(
                    re.findall(hw_counter_regex, build_in_vars[var])
                )
                subvariable_matches.update(re.findall(variable_regex, build_in_vars[var]))
        # process new found variables
        variable_matches = subvariable_matches - variable_matches
    return list(hw_counter_matches)


@demarcate
def perfmon_coalesce(
    pmc_files_list,
    perfmon_config,
    workload_dir,
    spatial_multiplexing,
    section_counters,
    mspec,
    arch,
):
    """Sort and bucket all related performance counters to minimize required application passes"""
    workload_perfmon_dir = workload_dir + "/perfmon"

    # Will be 2D array
    accumulate_counters = []

    normal_counters = OrderedDict()

    for fname in pmc_files_list:

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

            counters = m.group(1).split()

            if "SQ_ACCUM_PREV_HIRES" in counters:
                # Accumulate counters
                accumulate_counters.append(counters.copy())
            else:
                # Normal counters
                for ctr in counters:
                    # Remove me later:
                    # v1 and v2 don't support these counters
                    if not using_v3():
                        if ctr.startswith("SQ_INSTS_VALU_MFMA_F8") or ctr.startswith(
                            "SQ_INSTS_VALU_MFMA_MOPS_F8"
                        ):
                            continue

                    # Channel counter e.g. TCC_ATOMIC[0]
                    if "[" in ctr:

                        # FIXME:
                        # Remove channel number, append "_sum" so rocprof will
                        # sum the counters for us instead of specifying every
                        # channel.
                        channel = int(ctr.split("[")[1].split("]")[0])
                        if channel == 0:
                            counter_name = (
                                ctr.split("[")[0] + "_sum"
                                if using_v3()
                                else ctr.split("[")[0] + "_expand"
                            )

                            try:
                                normal_counters[counter_name] += 1
                            except:
                                normal_counters[counter_name] = 1
                    else:
                        try:
                            normal_counters[ctr] += 1
                        except:
                            normal_counters[ctr] = 1

    # De-duplicate. Remove accumulate counters from normal counters
    for accus in accumulate_counters:
        for accu in accus:
            if accu in normal_counters:
                del normal_counters[accu]

    # If section report filters have been provided, only collect counters necessary for those section reports
    # Remove _sum and _expand suffixes while matching
    def remove_suffixes(string):
        for suffix in ["_sum", "_expand"]:
            if string.endswith(suffix):
                string = string[: -len(suffix)]
                break
        return string

    section_counters = {remove_suffixes(counter) for counter in section_counters}
    ignored_counters = list()

    if section_counters:
        # Remove unnecessary normal counters
        for counter_name in list(normal_counters.keys()):
            if remove_suffixes(counter_name) not in section_counters:
                del normal_counters[counter_name]
                ignored_counters.append(counter_name)

        # Remove unnecessary accumulate counters
        filtered_accumlate_counters = list()
        for counters in accumulate_counters:
            if any(
                remove_suffixes(counter_name) in section_counters
                for counter_name in counters
            ):
                filtered_accumlate_counters.append(counters)
            else:
                ignored_counters.extend(counter_name)
        accumulate_counters = filtered_accumlate_counters

    if ignored_counters:
        console_log(
            f"Not collecting following counters per provided filter: {', '.join(ignored_counters)} "
        )

    # Throw error if no counters to be collected
    if len(normal_counters) == 0 and len(accumulate_counters) == 0:
        console_error(
            "profiling",
            "No performance counters to collect, please check the provided profiling filters",
        )

    output_files = []

    accu_file_count = 0
    # Each accumulate counter is in a different file
    for ctrs in accumulate_counters:

        ctr_name = ctrs[ctrs.index("SQ_ACCUM_PREV_HIRES") - 1]

        if using_v3():
            # v3 does not support SQ_ACCUM_PREV_HIRES. Instead we defined our own
            # counters in counter_defs.yaml that use the accumulate() function. These
            # use the name of the accumulate counter with _ACCUM appended to them.
            ctrs.remove("SQ_ACCUM_PREV_HIRES")

            accum_name = ctr_name + "_ACCUM"

            ctrs.append(accum_name)

        # Use the name of the accumulate counter as the file name
        output_files.append(CounterFile(ctr_name, perfmon_config))
        for ctr in ctrs:
            output_files[-1].add(ctr)
        accu_file_count += 1

    file_count = 0
    for ctr in normal_counters.keys():

        # Add counter to first file that has room
        added = False
        for f in output_files:
            if f.add(ctr):
                added = True
                break

        # All files are full, create a new file
        if not added:
            output_files.append(
                CounterFile("pmc_perf_{}.txt".format(file_count), perfmon_config)
            )
            file_count += 1
            output_files[-1].add(ctr)

    console_debug("profiling", "perfmon_coalesce file_count %s" % file_count)

    # TODO: rewrite the above logic for spatial_multiplexing later
    if spatial_multiplexing:

        # TODO: more error checking
        if len(spatial_multiplexing) != 3:
            console_error(
                "profiling",
                "multiplexing need provide node_idx node_count and gpu_count",
            )

        node_idx = int(spatial_multiplexing[0])
        node_count = int(spatial_multiplexing[1])
        gpu_count = int(spatial_multiplexing[2])

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
        for f in output_files:
            file_name_txt = str(Path(workload_perfmon_dir).joinpath(f.file_name_txt))
            file_name_yaml = str(Path(workload_perfmon_dir).joinpath(f.file_name_yaml))

            pmc = []
            for block_name in f.blocks.keys():
                if block_name == "TCC":
                    if using_v3():
                        # TODO: implement mehcanisim for muti channel TCC counter for v3
                        # Expand and interleve the TCC channel counters
                        # e.g.  TCC_HIT[0] TCC_ATOMIC[0] ... TCC_HIT[1] TCC_ATOMIC[1] ...
                        from utils.specs import total_xcds

                        channel_counters = []
                        xcds = total_xcds(mspec.gpu_model, mspec.compute_partition)
                        tcc_channel_per_xcd = int(mspec._l2_banks)

                        for ctr in f.blocks[block_name].elements:
                            if "_sum" in ctr:
                                channel_counters.append(ctr.split("_sum")[0])
                                pmc.append(ctr)

                        channel_counters = (
                            pd.Series(channel_counters).drop_duplicates().to_list()
                        )

                        lock = threading.Lock()

                        def generate_yaml_config_per_pmc(
                            raw_counter_name,
                            tcc_counter_1d_index,
                            xcd_index,
                            channel_index,
                        ):
                            # Ensure that the keys exist before trying to assign values
                            yaml_data = {}
                            if tcc_counter_1d_index not in yaml_data:
                                yaml_data[tcc_counter_1d_index] = {
                                    "architectures": {},
                                    "description": "",
                                }

                            if (
                                arch
                                not in yaml_data[tcc_counter_1d_index]["architectures"]
                            ):
                                yaml_data[tcc_counter_1d_index]["architectures"][
                                    arch
                                ] = {}

                            yaml_data[tcc_counter_1d_index]["architectures"][arch][
                                "expression"
                            ] = "select({},[DIMENSION_XCC=[{}], DIMENSION_INSTANCE=[{}]])".format(
                                raw_counter_name, xcd_index, channel_index
                            )
                            yaml_data[tcc_counter_1d_index]["description"] = (
                                "{} on {}th XCC and {}th channel".format(
                                    raw_counter_name, xcd_index, channel_index
                                )
                            )

                            lock.acquire()
                            pmc.append(tcc_counter_1d_index)
                            with open(file_name_yaml, "a") as file_yaml:
                                yaml.dump(
                                    yaml_data,
                                    file_yaml,
                                    default_flow_style=False,
                                    allow_unicode=True,
                                )
                            lock.release()

                        threads_edit_yaml = []

                        for i in range(0, xcds):
                            for j in range(0, tcc_channel_per_xcd):
                                for c in channel_counters:
                                    tcc_counter_1d_index = "{}[{}]".format(
                                        c, (i * tcc_channel_per_xcd) + j
                                    )

                                    thread = threading.Thread(
                                        target=generate_yaml_config_per_pmc,
                                        args=[c, tcc_counter_1d_index, i, j],
                                    )
                                    threads_edit_yaml.append(thread)
                                    thread.start()

                        for thread in threads_edit_yaml:
                            thread.join()

                        # Handle the rest of the TCC counters
                        for ctr in f.blocks[block_name].elements:
                            if "_expand" not in ctr and "_sum" not in ctr:
                                pmc.append(ctr)

                    else:
                        # Expand and interleve the TCC channel counters
                        # e.g.  TCC_HIT[0] TCC_ATOMIC[0] ... TCC_HIT[1] TCC_ATOMIC[1] ...
                        channel_counters = []
                        for ctr in f.blocks[block_name].elements:
                            if "_expand" in ctr:
                                channel_counters.append(ctr.split("_expand")[0])
                        for i in range(0, perfmon_config["TCC_channels"]):
                            for c in channel_counters:
                                pmc.append("{}[{}]".format(c, i))
                        # Handle the rest of the TCC counters
                        for ctr in f.blocks[block_name].elements:
                            if "_expand" not in ctr:
                                pmc.append(ctr)

                else:
                    if using_v3():
                        yaml_global_config_dir = (
                            get_default_accumulate_counter_file_ymal()
                        )
                        with open(yaml_global_config_dir, "r") as file_read:
                            with open(file_name_yaml, "a") as file_out:
                                dic_read = yaml.safe_load(file_read)
                                for ctr in f.blocks[block_name].elements:
                                    if ctr in dic_read:
                                        section_to_dump = {ctr: dic_read[ctr]}
                                        yaml.dump(
                                            section_to_dump,
                                            file_out,
                                            default_flow_style=False,
                                            allow_unicode=True,
                                        )
                                    pmc.append(ctr)

                    else:
                        for ctr in f.blocks[block_name].elements:
                            pmc.append(ctr)

            stext = "pmc: " + " ".join(pmc)

            # Write counters to file
            fd = open(file_name_txt, "w")
            fd.write(stext + "\n\n")
            fd.write("gpu:\n")
            fd.write("range:\n")
            fd.write("kernel:\n")
            fd.close()

    # Add a timestamp file
    # TODO: Does v3 need this?
    if not using_v3():
        fd = open(str(Path(workload_perfmon_dir).joinpath("timestamps.txt")), "w")
        fd.write("pmc:\n\n")
        fd.write("gpu:\n")
        fd.write("range:\n")
        fd.write("kernel:\n")
        fd.close()
