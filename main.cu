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
#include "material.h"

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


// Turning the followin recursive code to itereative with a limit of 50
__device__ vec3 color(const ray& r, hittable **d_world, curandState* local_rand_state) {
    ray curr_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);

    for (int i{ 1 }; i <= 50; i++) {
        hit_record rec;
        // Diffusion with attenuation 0.5
        // shadow acne removal by making t_min as 0.001
        if ((*d_world)->hit(curr_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(curr_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation = cur_attenuation * attenuation;
                curr_ray = scattered;
            }
            else {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(curr_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.6, 1.0);
            return cur_attenuation * c;
        }
    }

    return vec3(0.0, 0.0, 0.0); //exceeeds recursion limit
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
        col += color(r, d_world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(samples_per_pix);

    // Gamma correction
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);

    fb[pixel_index] = col;
}

__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(vec3(0, 0, -1), 0.5,
                               new lambertian(vec3(0.8, 0.3, 0.3)));
        d_list[1] = new sphere(vec3(0, -100.5, -1), 100,
                               new lambertian(vec3(0.8, 0.8, 0.0)));
        d_list[2] = new sphere(vec3(1, 0, -1), 0.5,
                               new metal(vec3(0.8, 0.6, 0.2), 1.0));
        d_list[3] = new sphere(vec3(-1, 0, -1), 0.5,
                               new metal(vec3(0.8, 0.8, 0.8), 0.3));
        *(d_world) = new hittable_list(d_list, 4);
        *d_camera = new camera();
    }
}

__global__ void free_world(hittable** d_list, hittable** d_world, camera** d_camera) {
    for (int i{ 0 }; i < 4; i++) {
        delete ((sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
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
    checkCudaErrors(cudaMalloc((void**)&d_list, 4 * sizeof(hittable*)));

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
    render_init << <blocks, threads >> > (image_width, image_height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
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