/**
 * @file sparse_log2_quant_plugin.cpp
 * @brief TensorRT Plugin 实现文件 for SparseLog2Quant.
 */

#include "sparse_log2_quant_plugin.h"
#include <cstring>
#include <cstdio>
#include <limits>

// ── SparseLog2QuantPlugin 实现 ─────────────────────────────────────────────────

SparseLog2QuantPlugin::SparseLog2QuantPlugin(
    std::vector<float> log2_base, bool per_channel
) : mLog2Base(std::move(log2_base)),
    mPerChannel(per_channel),
    mLog2BaseDevice(nullptr),
    mCachedC(0),
    mLog2BaseLen(static_cast<int32_t>(mLog2Base.size())) {
}

SparseLog2QuantPlugin::SparseLog2QuantPlugin(const void* data, size_t length)
    : mPerChannel(false), mLog2BaseDevice(nullptr), mCachedC(0) {
    // 头部固定开销：size_t(base_size) + int32_t(per_channel)
    constexpr size_t kHeaderSize = sizeof(size_t) + sizeof(int32_t);
    if (data == nullptr || length < kHeaderSize) {
        return;
    }

    const char* buf = static_cast<const char*>(data);

    // 反序列化 log2_base
    size_t base_size = 0;
    std::memcpy(&base_size, buf, sizeof(size_t));
    buf += sizeof(size_t);

    // base_size 来自不可信输入：先校验浮动区长度，避免 resize/memcpy 越界或 bad_alloc
    const size_t payload_bytes = length - kHeaderSize;
    if (base_size > payload_bytes / sizeof(float)) {
        return;
    }

    mLog2Base.resize(base_size);
    if (base_size > 0) {
        std::memcpy(mLog2Base.data(), buf, base_size * sizeof(float));
        buf += base_size * sizeof(float);
    }

    // 反序列化 per_channel
    int32_t per_ch_int = 0;
    std::memcpy(&per_ch_int, buf, sizeof(int32_t));
    mPerChannel = (per_ch_int != 0);

    mLog2BaseDevice = nullptr;
    mCachedC = static_cast<int32_t>(base_size);
    mLog2BaseLen = static_cast<int32_t>(base_size);
}

SparseLog2QuantPlugin::~SparseLog2QuantPlugin() {
    terminate();
}

nvinfer1::DimsExprs SparseLog2QuantPlugin::getOutputDimensions(
    int32_t outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int32_t nbInputs,
    nvinfer1::IExprBuilder& exprBuilder
) noexcept {
    // 输出形状与输入完全相同
    return inputs[0];
}

bool SparseLog2QuantPlugin::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int32_t nbInputs,
    int32_t nbOutputs
) noexcept {
    // 仅支持 FP16 + kLINEAR
    // pos 0: input, pos 1: output
    if (pos < nbInputs) {
        // 输入格式检查
        return inOut[pos].type == nvinfer1::DataType::kHALF &&
               inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    } else {
        // 输出格式检查（与输入相同）
        return inOut[pos].type == inOut[0].type &&
               inOut[pos].format == inOut[0].format;
    }
}

void SparseLog2QuantPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int32_t nbOutputs
) noexcept {
    // 从输入维度获取通道数
    const auto& dims = in[0].desc.dims;
    int32_t nb_dims = dims.nbDims;

    if (nb_dims >= 2) {
        // 假设 NCHW 或 NC 格式，通道维是第 1 维
        mCachedC = dims.d[1];
    } else {
        mCachedC = 1;
    }

    // 分配并拷贝 log2_base 到 GPU
    if (mLog2BaseDevice == nullptr && !mLog2Base.empty()) {
        const size_t base_bytes = mLog2Base.size() * sizeof(float);
        if (cudaMalloc(&mLog2BaseDevice, base_bytes) != cudaSuccess) {
            mLog2BaseDevice = nullptr;
            return;
        }
        if (cudaMemcpy(
                mLog2BaseDevice,
                mLog2Base.data(),
                base_bytes,
                cudaMemcpyHostToDevice
            ) != cudaSuccess) {
            cudaFree(mLog2BaseDevice);
            mLog2BaseDevice = nullptr;
        }
    }
}

size_t SparseLog2QuantPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int32_t nbOutputs
) const noexcept {
    // 不需要额外工作空间
    return 0;
}

int32_t SparseLog2QuantPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream
) noexcept {
    // 输入契约校验：指针、维度、dtype、GPU base 缓冲区
    if (inputDesc == nullptr || outputDesc == nullptr ||
        inputs == nullptr || outputs == nullptr ||
        inputs[0] == nullptr || outputs[0] == nullptr ||
        mLog2BaseDevice == nullptr || mLog2Base.empty()) {
        return -1;
    }
    if (inputDesc[0].type != nvinfer1::DataType::kHALF ||
        outputDesc[0].type != nvinfer1::DataType::kHALF) {
        return -1;
    }

    // 获取输入维度信息
    const auto& dims = inputDesc[0].dims;
    int32_t nb_dims = dims.nbDims;

    // 准备维度参数（按值传递给 kernel）
    int32_t dim0 = (nb_dims > 0) ? dims.d[0] : 1;  // N
    int32_t dim1 = (nb_dims > 1) ? dims.d[1] : 1;  // C
    int32_t dim2 = (nb_dims > 2) ? dims.d[2] : 1;  // H or D
    int32_t dim3 = (nb_dims > 3) ? dims.d[3] : 1;  // W

    // 防止 dims.d[] (int64) 截断为 int32 时溢出/变负
    if (dim0 < 0 || dim1 <= 0 || dim2 <= 0 || dim3 <= 0 ||
        dims.d[0] > std::numeric_limits<int32_t>::max() ||
        dims.d[1] > std::numeric_limits<int32_t>::max() ||
        dims.d[2] > std::numeric_limits<int32_t>::max() ||
        dims.d[3] > std::numeric_limits<int32_t>::max()) {
        return -1;
    }

    // per-channel 模式下 log2_base 长度必须等于通道数 C，否则 kernel 内 log2_base[c] 越界
    if (mPerChannel && mLog2BaseLen != dim1) {
        return -1;
    }

    // 调用 CUDA kernel
    launch_sparse_log2_quant(
        inputs[0],
        outputs[0],
        mLog2BaseDevice,
        dim0,
        dim1,
        dim2,
        dim3,
        nb_dims,
        mPerChannel,
        stream
    );

    // 检查 kernel launch 是否成功（异步错误在后续同步点暴露）
    if (cudaGetLastError() != cudaSuccess) {
        return -1;
    }

    return 0;  // 成功
}

nvinfer1::DataType SparseLog2QuantPlugin::getOutputDataType(
    int32_t index,
    const nvinfer1::DataType* inputTypes,
    int32_t nbInputs
) const noexcept {
    // 输出类型与输入相同
    return inputTypes[0];
}

// ── IPluginV2 基础接口 ─────────────────────────────────────────────────────

const char* SparseLog2QuantPlugin::getPluginType() const noexcept {
    return LOG2_PLUGIN_NAME;
}

const char* SparseLog2QuantPlugin::getPluginVersion() const noexcept {
    return LOG2_PLUGIN_VERSION;
}

int32_t SparseLog2QuantPlugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t SparseLog2QuantPlugin::initialize() noexcept {
    return 0;  // 成功
}

void SparseLog2QuantPlugin::terminate() noexcept {
    if (mLog2BaseDevice != nullptr) {
        cudaFree(mLog2BaseDevice);
        mLog2BaseDevice = nullptr;
    }
}

size_t SparseLog2QuantPlugin::getSerializationSize() const noexcept {
    // size_t (base_size) + float[] (log2_base) + int32_t (per_channel)
    return sizeof(size_t) + mLog2Base.size() * sizeof(float) + sizeof(int32_t);
}

void SparseLog2QuantPlugin::serialize(void* buffer) const noexcept {
    char* buf = static_cast<char*>(buffer);

    // 序列化 log2_base
    size_t base_size = mLog2Base.size();
    std::memcpy(buf, &base_size, sizeof(size_t));
    buf += sizeof(size_t);

    std::memcpy(buf, mLog2Base.data(), base_size * sizeof(float));
    buf += base_size * sizeof(float);

    // 序列化 per_channel
    int32_t per_ch_int = mPerChannel ? 1 : 0;
    std::memcpy(buf, &per_ch_int, sizeof(int32_t));
}

void SparseLog2QuantPlugin::destroy() noexcept {
    delete this;
}

void SparseLog2QuantPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const char* SparseLog2QuantPlugin::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

nvinfer1::IPluginV2DynamicExt* SparseLog2QuantPlugin::clone() const noexcept {
    auto* cloned = new SparseLog2QuantPlugin(mLog2Base, mPerChannel);

    // 如果当前实例已分配 GPU 内存，也需要为 clone 分配
    if (mLog2BaseDevice != nullptr) {
        const size_t base_bytes = mLog2Base.size() * sizeof(float);
        if (cudaMalloc(&cloned->mLog2BaseDevice, base_bytes) != cudaSuccess) {
            cloned->mLog2BaseDevice = nullptr;
        } else if (cudaMemcpy(
                       cloned->mLog2BaseDevice,
                       mLog2BaseDevice,
                       base_bytes,
                       cudaMemcpyDeviceToDevice
                   ) != cudaSuccess) {
            cudaFree(cloned->mLog2BaseDevice);
            cloned->mLog2BaseDevice = nullptr;
        }
    }

    cloned->mCachedC = mCachedC;
    cloned->mLog2BaseLen = mLog2BaseLen;
    cloned->mNamespace = mNamespace;

    return cloned;
}

// ── SparseLog2QuantPluginCreator 实现 ─────────────────────────────────────────

SparseLog2QuantPluginCreator::SparseLog2QuantPluginCreator() {
    // 定义 Plugin 字段
    mFields = {
        {"log2_base",   nullptr, nvinfer1::PluginFieldType::kFLOAT32, 0},
        {"per_channel", nullptr, nvinfer1::PluginFieldType::kINT32,   1},
    };
    mFC.nbFields = static_cast<int32_t>(mFields.size());
    mFC.fields   = mFields.data();
}

const char* SparseLog2QuantPluginCreator::getPluginName() const noexcept {
    return LOG2_PLUGIN_NAME;
}

const char* SparseLog2QuantPluginCreator::getPluginVersion() const noexcept {
    return LOG2_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection*
SparseLog2QuantPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

nvinfer1::IPluginV2* SparseLog2QuantPluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc
) noexcept {
    std::vector<float> log2_base;
    bool per_channel = false;

    if (fc == nullptr || fc->nbFields < 0 ||
        (fc->nbFields > 0 && fc->fields == nullptr)) {
        return nullptr;
    }

    // 解析 Plugin 字段
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        const auto& field = fc->fields[i];
        if (field.name == nullptr || field.data == nullptr) {
            return nullptr;
        }

        if (std::string(field.name) == "log2_base") {
            if (field.type != nvinfer1::PluginFieldType::kFLOAT32 ||
                field.length <= 0) {
                return nullptr;
            }
            const float* data = static_cast<const float*>(field.data);
            log2_base.assign(data, data + field.length);
        } else if (std::string(field.name) == "per_channel") {
            if (field.type != nvinfer1::PluginFieldType::kINT32 ||
                field.length != 1) {
                return nullptr;
            }
            per_channel = (*static_cast<const int32_t*>(field.data)) != 0;
        }
    }

    // 验证参数
    if (log2_base.empty()) {
        fprintf(stderr, "[SparseLog2QuantPluginCreator] ERROR: log2_base is empty\n");
        return nullptr;
    }

    return new SparseLog2QuantPlugin(std::move(log2_base), per_channel);
}

nvinfer1::IPluginV2* SparseLog2QuantPluginCreator::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength
) noexcept {
    constexpr size_t kHeaderSize = sizeof(size_t) + sizeof(int32_t);
    if (serialData == nullptr || serialLength < kHeaderSize) {
        return nullptr;
    }
    return new SparseLog2QuantPlugin(serialData, serialLength);
}

void SparseLog2QuantPluginCreator::setPluginNamespace(
    const char* pluginNamespace
) noexcept {
    mNamespace = pluginNamespace;
}

const char* SparseLog2QuantPluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

// ── 显式初始化函数（用于强制加载 .so 时执行全局构造函数）──────────────────────

// 供外部调用的显式初始化函数（仅用于确保 .so 被加载，不重复注册）
// 实际的 Plugin 注册由 REGISTER_TENSORRT_PLUGIN 宏自动完成
extern "C" {
    __attribute__((visibility("default")))
    void forceInitSparseLog2QuantPlugin() {
        // 此函数为空，仅用于在 ctypes/dlopen 时强制加载 .so
        // 实际的 Plugin 注册由 REGISTER_TENSORRT_PLUGIN 宏在全局构造时完成
        // 避免重复注册：不在这里调用 registerCreator
    }
}

// 使用 REGISTER_TENSORRT_PLUGIN 宏进行自动注册（标准做法）
REGISTER_TENSORRT_PLUGIN(SparseLog2QuantPluginCreator);
