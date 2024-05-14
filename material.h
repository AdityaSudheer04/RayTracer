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

__device__ vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
    auto cos_theta = fmin(dot(-uv, n), 1.0f);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0f - r_out_perp.length_squared())) * n;
    return r_out_parallel + r_out_perp;
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

class dielectric : public material {
public:
    __device__ dielectric(float refraction_index) : refraction_index(refraction_index) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, 
                curandState* local_rand_state)
        const override {
        attenuation = vec3(1.0, 1.0, 1.0);
        float ri = rec.front_face ? (1.0f / refraction_index) : refraction_index;

        vec3 unit_direction = unit_vector(r_in.direction());
        float cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0f);
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0f;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, ri) > curand_uniform(local_rand_state))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, ri);

        scattered = ray(rec.p, direction);
        return true;
    }

private:
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    float refraction_index;

    __device__ static float reflectance(float cosine, float refraction_index) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * pow((1.0f - cosine), 5);
    }
};


#endif