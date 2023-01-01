#pragma once
#define MAX_IMAGES 1024


#define FORCEINLINE __forceinline__

// Vector operations
#define dot4(v0,v1) v0.x * v1.x + \
                    v0.y * v1.y + \
                    v0.z * v1.z

#define matvecmul4P(m, v, out) \
out->x = \
m [0] * v.x + \
m [1] * v.y + \
m [2] * v.z + \
m [3]; \
out->y = \
m [4] * v.x + \
m [5] * v.y + \
m [6] * v.z + \
m [7]; \
out->z = \
m [8] * v.x  + \
m [9] * v.y  + \
m [10] * v.z + \
m [11];

#define matvecmul4(m, v, out) \
out->x = \
m [0] * v.x + \
m [1] * v.y + \
m [2] * v.z; \
out->y = \
m [3] * v.x + \
m [4] * v.y + \
m [5] * v.z; \
out->z = \
m [6] * v.x + \
m [7] * v.y + \
m [8] * v.z;
