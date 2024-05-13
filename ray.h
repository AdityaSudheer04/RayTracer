#ifndef RAY_H
#define RAY_H

// class for ray where ray can be written as
// P(t) = A + t*b
// A is the origin, b is the direction, t is the variable

#include "vec3.h"
#include<cuda_runtime.h>

class ray {
public:

    __device__ ray () {

    }

    __device__ ray (const point3& origin, const vec3& direction)
        :  m_dir{direction}, m_orig{origin}
    {

    }

    __device__ const point3& origin() const { return m_orig; }
    __device__ const vec3& direction() const{ return m_dir; }

    __device__ point3 at(float t) const {
        return m_orig + t*m_dir;
    }

private:
    vec3 m_dir;
    point3 m_orig;
};

#endif