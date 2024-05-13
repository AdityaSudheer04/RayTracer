#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ float hit_sphere(const vec3& center, float radius, const ray& r) {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(r.direction(), oc);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0) return -1.0f;
    else return ( ( - b - sqrt(discriminant)) / (2.0f * a) );
}

__device__ vec3 color(const ray& r, hittable **d_world) {
    hit_record rec;
    if ((*d_world)->hit(r, 0.0, FLT_MAX, rec)) {
        return 0.5f * vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
    }
    vec3 unit_direction = unit_vector(r.direction());
    float a = 0.5f * (unit_direction.y() + 1.0f);   //Writing f required as doube precision is default
    return (1.0f - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.3, 0.3, 1.0);
}   

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* fb, int max_x, int max_y, int samples_per_pix,
                       camera** cam, hittable **d_world, curandState* rand_state) {
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < samples_per_pix; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v);
        col += color(r, d_world);
    }
    fb[pixel_index] = col / float(samples_per_pix);
}

__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list) = new sphere(vec3(0, 0, -2), 0.5);
        *(d_list + 1) = new sphere(vec3(0, -100.5, -2), 100);
        *(d_list + 2) = new sphere(vec3(-2, 0, -3), 0.5);
        *(d_world) = new hittable_list(d_list, 3);
        *d_camera = new camera();
    }
}

__global__ void free_world(hittable** d_list, hittable** d_world, camera** d_camera) {
    delete* (d_list);
    delete* (d_list + 1);
    delete* (d_list + 2);
    delete* d_world;
    delete* d_camera;
}

int main() {
    int image_width = 1200;
    int image_height = 600;
    int samples_per_pix = 100;
    int tx = 16;
    int ty = 16;

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = image_width * image_height;
    size_t fb_size = num_pixels * sizeof(vec3);

    // allocate Frame Buffer
    // Frame Buffer is a vec3 array that contains all the values of RGB fo each pixel
    // from top left to bottom right
    vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // allocate random state
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

    // make our world of hittables and camera
    hittable** d_list;  // d prefix for device only
    checkCudaErrors(cudaMalloc((void**)&d_list, 3 * sizeof(hittable*)));

    hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));

    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));

    create_world << <1, 1 >> > (d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    
    clock_t start, stop;
    start = clock();

    // Render our buffer
    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);
    render << <blocks, threads >> > (fb, image_width, image_height, 
                                    samples_per_pix,
                                    d_camera,
                                    d_world,
                                    d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
    for (int j = image_height - 1; j >= 0; j--) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j * image_width + i;
            int ir = int(255.99 * fb[pixel_index].x());
            int ig = int(255.99 * fb[pixel_index].y());
            int ib = int(255.99 * fb[pixel_index].z());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world << <1, 1 >> > (d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}