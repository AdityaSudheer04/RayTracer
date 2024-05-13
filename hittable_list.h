#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"


class hittable_list : public hittable {

public:
     hittable **objects;
     int list_size;

    __device__ hittable_list () {}
    __device__ hittable_list(hittable** list, int n) { objects = list, list_size = n; }

    /*void clear() { objects.clear(); }

    void add( object){
        objects.push_back(object);
    }*/

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override {
        
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max; 


        for (int i{ 0 }; i < list_size; i++) {
            if(objects[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
};

#endif