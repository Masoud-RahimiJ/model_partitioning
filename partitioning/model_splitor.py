from enum import Enum

class Operation(Enum):
    SPLIT = 1
    MERGE = 2
    

class ModelSplitor:
    def __init__(self, layers_size, layers_execution_time, network_throughput, download_delay, num_cpu_threads=2, additional_memmory_available=0):
        self.layers_count = len(layers_size)
        self.layers_size = layers_size
        self.layers_execution_time = layers_execution_time
        self.network_throughput = network_throughput
        self.download_delay = download_delay
        self.layers_download_time = [self.layers_size[idx] / self.network_throughput for idx in range(self.layers_count)]
        self.cpu_capasity_for_previous_layers = [0] * (self.layers_count + 1)
        self.layers_startup_time = [0] * (self.layers_count + 1)
        self.num_cpu_threads = num_cpu_threads
        self.additional_memmory_available = additional_memmory_available
        
    def calculate_previous_layer_metrics_without_merging(self, idx, previous_partition_download_time, previous_partition_execution_time):
        e_previous_layer_minus_d_layer = max(0, self.layers_execution_time[idx-1] - previous_partition_download_time)
        previous_layer_startup_time = self.layers_startup_time[idx+1] + previous_partition_download_time + \
            self.layers_download_time[idx-1] + self.download_delay + \
            max(0, self.layers_execution_time[idx-1] - previous_partition_download_time - self.cpu_capasity_for_previous_layers[idx+1]) + \
            max(0, previous_partition_execution_time - max(0, self.cpu_capasity_for_previous_layers[idx+1] - e_previous_layer_minus_d_layer))
        cpu_capacity_for_previous_layer = (self.layers_download_time[idx-1] + self.download_delay)*self.num_cpu_threads + \
            max(0, previous_partition_download_time*self.num_cpu_threads - self.layers_execution_time[idx-1]) + \
            max(0, self.cpu_capasity_for_previous_layers[idx+1] - previous_partition_execution_time - e_previous_layer_minus_d_layer)
        return previous_layer_startup_time, cpu_capacity_for_previous_layer
    
    def calculate_previous_layer_metrics_with_merging(self, idx, previous_partition_download_time, previous_partition_execution_time):
        previous_layer_startup_time = self.layers_startup_time[idx+1] + previous_partition_download_time + self.layers_download_time[idx-1] + \
            max(0, self.layers_execution_time[idx-1] + previous_partition_execution_time - self.cpu_capasity_for_previous_layers[idx+1])
        cpu_capacity_for_previous_layer = (self.layers_download_time[idx-1] + previous_partition_download_time)*self.num_cpu_threads + \
            max(0, self.cpu_capasity_for_previous_layers[idx+1] - self.layers_execution_time[idx-1] - previous_partition_execution_time )
        return previous_layer_startup_time, cpu_capacity_for_previous_layer                
                    
    def add_spit_operation(self, idx, sts, ccs, free_memmory_space):
        self.layers_startup_time[idx-1] = sts
        self.cpu_capasity_for_previous_layers[idx-1] = ccs
        partitions = self.calculate_optimized_partitions(idx-1, self.download_delay + self.layers_download_time[idx-1], self.layers_execution_time[idx-1], self.layers_size[idx-1], free_memmory_space+self.layers_size[idx-1])
        partitions.append(Operation.SPLIT)
        return partitions
    
    def add_merge_operation(self, idx, previous_partition_download_time, previous_partition_execution_time, stm, ccm, previous_partition_size, free_memmory_space):
        self.layers_startup_time[idx-1] = stm
        self.cpu_capasity_for_previous_layers[idx-1] = ccm
        partitions = self.calculate_optimized_partitions(idx-1, previous_partition_download_time + self.layers_download_time[idx-1], self.layers_execution_time[idx-1] + previous_partition_execution_time, self.layers_execution_time[idx-1], previous_partition_size+self.layers_size[idx-1], free_memmory_space+self.layers_size[idx-1])
        partitions.append(Operation.MERGE)
        return partitions
    
    def calculate_optimized_partitions(self, idx, previous_partition_download_time, previous_partition_execution_time, previous_partition_size, free_memmory_space):
        if idx == 0:
            return []
        sts, ccs = self.calculate_previous_layer_metrics_without_merging(idx, previous_partition_download_time, previous_partition_execution_time)
        stm, ccm = self.calculate_previous_layer_metrics_with_merging(idx, previous_partition_download_time, previous_partition_execution_time)
        if free_memmory_space < previous_partition_size + self.layers_size[idx-1]*2:
            return self.add_spit_operation(idx, sts, ccs, free_memmory_space)
        elif sts <= stm:
            if stm - sts >=  ccm - ccs:
                return self.add_spit_operation(idx, sts, ccs)
            else:
                partitions_with_merge = self.add_merge_operation(idx, previous_partition_download_time, previous_partition_execution_time, stm, ccm, previous_partition_size, free_memmory_space)
                if self.cpu_capasity_for_previous_layers[1] > ccm - ccs - (stm - sts):
                    startup_time_with_merging = self.layers_startup_time[0]
                    partitions_with_split = self.add_spit_operation(idx, sts, ccs)
                    return partitions_with_split if self.layers_startup_time[0] <= startup_time_with_merging else partitions_with_merge
                else:
                    return partitions_with_merge
        else:
            if sts - stm >=  ccs - ccm:
                return self.add_merge_operation(idx, previous_partition_download_time, previous_partition_execution_time, stm, ccm, previous_partition_size, free_memmory_space)
            else:
                partitions_with_split = self.add_spit_operation(idx, sts, ccs)
                if self.cpu_capasity_for_previous_layers[1] > ccs - ccm - (sts - stm):
                    startup_time_with_split = self.layers_startup_time[0]
                    partitions_with_merge = self.add_merge_operation(idx, previous_partition_download_time, previous_partition_execution_time, stm, ccm, previous_partition_size, free_memmory_space)
                    return partitions_with_merge if self.layers_startup_time[0] <= startup_time_with_split else partitions_with_split
                else:
                    return partitions_with_split
                
    def find_optimized_partitions(self):
        self.cpu_capasity_for_previous_layers[-2] = self.download_delay + self.layers_download_time[-1]
        self.layers_startup_time[-2] = self.download_delay + self.layers_download_time[-1] + self.layers_execution_time[-1]
        partitions = self.calculate_optimized_partitions(
            self.layers_count-1,
            self.download_delay + self.layers_download_time[-1],
            self.layers_execution_time[-1],
            self.layers_size[-1],
            self.layers_size[-1] + self.additional_memmory_available
            )
        self.simulate_partition_startup(partitions)
        return partitions
        
    def simulate_partition_startup(self, partitions):
        self.cpu_capasity_for_previous_layers = [0] * (self.layers_count + 1)
        self.layers_startup_time = [0] * (self.layers_count + 1)
        self.cpu_capasity_for_previous_layers[-2] = self.download_delay + self.layers_download_time[-1]
        self.layers_startup_time[-2] = self.download_delay + self.layers_download_time[-1] + self.layers_execution_time[-1]
        previous_partition_download_time = self.download_delay + self.layers_download_time[-1]
        previous_partition_execution_time = self.layers_execution_time[-1]
        max_memmory_usage=sum(self.layers_size) + self.layers_size[-1]
        for idx in range(self.layers_count-1, 0, -1):
            if partitions[idx-1] == Operation.SPLIT:
                sts, ccs = self.calculate_previous_layer_metrics_without_merging(idx, previous_partition_download_time, previous_partition_execution_time)
                self.layers_startup_time[idx-1] = sts
                self.cpu_capasity_for_previous_layers[idx-1] = ccs
                previous_partition_download_time = self.layers_download_time[idx-1] + self.download_delay
                previous_partition_execution_time = self.layers_execution_time[idx-1]
            else:
                stm, ccm = self.calculate_previous_layer_metrics_with_merging(idx, previous_partition_download_time, previous_partition_execution_time)
                self.layers_startup_time[idx-1] = stm
                self.cpu_capasity_for_previous_layers[idx-1] = ccm
                previous_partition_download_time += self.layers_download_time[idx-1]
                previous_partition_execution_time += self.layers_execution_time[idx-1]
            max_memmory_usage = max(max_memmory_usage, sum(self.layers_size[:idx]) + self.layers_size[idx])
