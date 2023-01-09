#include "globalstate.h"
#include "point_cloud_list.h"

void dump_gpu_memory();
int runcuda(GlobalState & gs, PointCloudList & pc_list, int num_views);
