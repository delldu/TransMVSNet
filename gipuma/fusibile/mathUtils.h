/*
 * some math helper functions
 */

#pragma once

#ifndef M_PI
#define M_PI    3.14159265358979323846f
#endif
#define M_PI_float    3.14159265358979323846f

//rounding of positive float values
#if defined(_WIN32)
static float roundf ( float val ) {
    return floor ( val+0.5f );
};
#endif
