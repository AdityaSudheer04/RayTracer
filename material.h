#ifndef MATERIAL_H
#define MATERIAL_H

#include "ray.h"
#include "hittable.h"
#include <curand.h>
#include <curand_kernel.h>

class hit_record;

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}

class material {
public:
    //virtual ~material() = default;

    __device__ virtual bool scatter(const ray& r_in,const hit_record& rec, vec3& attenuation, ray& scattered
                                    , curandState* local_rand_state) const {
        return false;
    }
    
};

class lambertian : public material {
public:
    __device__ lambertian(const vec3& albedo) : albedo{albedo} {

    }

    __device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered,
                            curandState* local_rand_state)
    const override {
        vec3 scatter_direction = rec.normal + unit_vector(random_in_unit_sphere(local_rand_state));

        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;
        
        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }
private:
    vec3 albedo;
};

class metal : public material {
public:
    __device__ metal(const vec3& albedo, float fuzz) : albedo{albedo}, fuzz(fuzz < 1 ? fuzz : 1) {

    }

    __device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered,
                curandState* local_rand_state)
    const override {
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        reflected = unit_vector(reflected) + (fuzz * random_in_unit_sphere(local_rand_state));
        scattered = ray(rec.p, reflected);
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0.0f);
    }
private:
    vec3 albedo;
    float fuzz;
};


#endif