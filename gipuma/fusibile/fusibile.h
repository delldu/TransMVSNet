#include "globalstate.h"
#include "point_cloud_list.h"

void dump_gpu_memory();
int run_cuda(GlobalState & gs, PointCloudList & pc_list, int num_views);
