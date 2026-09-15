// BEVPoolV2 CUDA Kernel
// Matching BEVFusion's existing interval-sum kernel in bev_pool_cuda.cu
//
// Input format:
//   - x: [N, C] flattened features after outer product
//   - geom_feats: [N, 4] voxel coordinates (x, y, z, batch)
//   - interval_starts: [M] interval start indices
//   - interval_lengths: [M] interval lengths
//
// Output: [B, D, H, W, C] BEV features (D=Z, H=X, W=Y, matching Python call order)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

/*
  BEV Pooling V2 Kernel (interval-sum style)
  
  This matches BEVFusion's kernel in mmdet3d/ops/bev_pool/src/bev_pool_cuda.cu
  
  Args:
    b                : batch size
    d                : depth of the feature map (Z)
    h                : height of pooled feature map (Y)
    w                : width of pooled feature map (X)
    n_intervals      : number of unique BEV grids to sum
    c                : number of channels
    x                : input features, FloatTensor[N, C]
    geom_feats       : input coordinates, IntTensor[N, 4] (x, y, z, batch)
    interval_starts  : starting position for pooled point, IntTensor[M]
    interval_lengths : how many points in each pooled point, IntTensor[M]
    out              : output features, FloatTensor[B, D, H, W, C]
*/
__global__ void __launch_bounds__(256) bev_pool_v2_kernel(
    int b, int d, int h, int w, int n_points, int n_intervals, int c,
    const float* __restrict__ x,
    const int* __restrict__ geom_feats,
    const int* __restrict__ interval_starts,
    const int* __restrict__ interval_lengths,
    float* __restrict__ out)
{
    long long total_threads = static_cast<long long>(n_intervals) * c;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total_threads;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        int index = static_cast<int>(idx / c);
        int cur_c = static_cast<int>(idx % c);

        int interval_start = interval_starts[index];
        int interval_length = interval_lengths[index];
        long long interval_end = static_cast<long long>(interval_start) + interval_length;
        if (interval_start < 0 || interval_length < 0 || interval_end > n_points) continue;
        if (interval_length == 0) continue;

        const int* cur_geom_feats = geom_feats + static_cast<long long>(interval_start) * 4;
        int bev_x = cur_geom_feats[0];
        int bev_y = cur_geom_feats[1];
        int bev_z = cur_geom_feats[2];
        int batch_idx = cur_geom_feats[3];
        if (bev_x < 0 || bev_x >= h || bev_y < 0 || bev_y >= w ||
            bev_z < 0 || bev_z >= d || batch_idx < 0 || batch_idx >= b) continue;

        long long output_offset = static_cast<long long>(batch_idx) * d * h * w * c +
            static_cast<long long>(bev_z) * h * w * c +
            static_cast<long long>(bev_x) * w * c +
            static_cast<long long>(bev_y) * c + cur_c;
        float* cur_out = out + output_offset;

        const float* cur_x = x + static_cast<long long>(interval_start) * c + cur_c;
        float psum = 0.0f;
        for (int i = 0; i < interval_length; i++) {
            psum += cur_x[static_cast<long long>(i) * c];
        }

        *cur_out = psum;
    }
}

// C interface for Plugin to call
extern "C" void launch_bev_pool_v2(
    int b, int d, int h, int w, int n_points, int n_intervals, int c,
    const float* x,
    const int* geom_feats,
    const int* interval_starts,
    const int* interval_lengths,
    float* out, cudaStream_t stream)
{
    long long total_threads = static_cast<long long>(n_intervals) * c;
    if (total_threads <= 0 || n_points < 0 || b <= 0 || d <= 0 || h <= 0 || w <= 0 || c <= 0) {
        return;
    }
    const int block_size = 256;
    long long grid_size_64 = (total_threads + block_size - 1) / block_size;
    int grid_size = static_cast<int>(grid_size_64 > 65535LL ? 65535LL : grid_size_64);

    bev_pool_v2_kernel<<<grid_size, block_size, 0, stream>>>(
        b, d, h, w, n_points, n_intervals, c,
        x, geom_feats, interval_starts, interval_lengths, out);
}
