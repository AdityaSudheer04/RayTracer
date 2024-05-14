#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vec3.h"

class sphere : public hittable {
public:
    __device__ sphere(const point3& center, float radius, material* mat) 
        : center{ center }, radius{ radius }, mat_ptr{mat} {

    }

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override {
            //Gettin t for the normal when ray hits using equation 
        // (d.d)t^2 - (2*d*(C - Q))t + (C - Q)*(C - Q) - r^2 = 0
        // d is direction, C is center , Q is ray origin

        vec3 oc = center - r.origin();
        float a = dot(r.direction(),r.direction());

        //Taking h = -b/2
        float h = dot(r.direction(),oc);
        float c = oc.length_squared() - radius*radius;

        float discriminant = h*h - a*c;
        
        if(discriminant < 0){
            return false;
        }

        float sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        float root = (h - sqrtd) / a;
        if ( !(t_min < root && root < t_max) ) {
            root = (h + sqrtd) / a;
            if ( !(t_min < root && root < t_max) )
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat_ptr = mat_ptr;
        
        return true;
    }
    material* mat_ptr;
private:
    point3 center;
    float radius;
};


#endif