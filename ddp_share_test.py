import torch
import torch.multiprocessing as mp

from torch.multiprocessing import Manager
import torch.distributed as dist
from torch.distributed import launch
# manager = Manager()
# feature_dict = manager.dict()

class SharedFeaturePool:
    def __init__(self, local_rank):
        self.feature_pool = dict()
        self.local_rank = local_rank
        if self.local_rank == 0:
            self.synthesize()
            dist.barrier()
        else:
            dist.barrier() # Wait synthesize on main process
            self.receive()
            # dist.broadcast_object_list(obj, src=0)
        obj = [(k, v) for k, v in self.feature_pool.items()]
        dist.broadcast_object_list(obj, src=0)
        if self.local_rank != 0:
            for k, v in obj:
                self.feature_pool[k] = v
        dist.barrier()

    def synthesize(self):
        for i in range(1000):
            self.feature_pool[i] = torch.randn([100]).share_memory_()

    def receive(self):
        for i in range(1000):
            self.feature_pool[i] = None

    def __getitem__(self, key):
        return self.feature_pool[key]

if __name__ == "__main__":
    dist.init_process_group("gloo")
    local_rank = dist.get_rank()
    local_size = dist.get_world_size()
    mp.set_start_method("spawn")
    mp.set_sharing_strategy("file_system")

    

    feature_pool = SharedFeaturePool(local_rank)

    print(local_rank, feature_pool[122])
    