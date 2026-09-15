// BEVPoolV2 TensorRT Plugin Implementation
// Matching BEVFusion's bev_pool interface

#include "bev_pool_v2_plugin.h"
#include <cstring>
#include <iostream>
#include <limits>

// Constructor from ONNX attributes
// B: batch size, D: depth (Z), H: height (Y), W: width (X)
BEVPoolV2Plugin::BEVPoolV2Plugin(int B, int D, int H, int W)
    : mB(B), mD(D), mH(H), mW(W) {}

// Constructor from serialized data
BEVPoolV2Plugin::BEVPoolV2Plugin(const void* data, size_t length)
    : mB(0), mD(0), mH(0), mW(0) {
    if (data == nullptr || length < 4 * sizeof(int)) {
        return;
    }
    const char* buf = static_cast<const char*>(data);
    memcpy(&mB, buf, sizeof(int)); buf += sizeof(int);
    memcpy(&mD, buf, sizeof(int)); buf += sizeof(int);
    memcpy(&mH, buf, sizeof(int)); buf += sizeof(int);
    memcpy(&mW, buf, sizeof(int));
    if (mB <= 0 || mD <= 0 || mH <= 0 || mW <= 0) {
        mB = mD = mH = mW = 0;
    }
}

BEVPoolV2Plugin::~BEVPoolV2Plugin() {}

bool BEVPoolV2Plugin::validate_input_contract(
    const nvinfer1::PluginTensorDesc* const inputDesc,
    int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* const outputDesc,
    int32_t nbOutputs,
    const void* const* const inputs,
    void* const* const outputs) const noexcept {
    if (inputDesc == nullptr || outputDesc == nullptr ||
        inputs == nullptr || outputs == nullptr || nbInputs != 4 || nbOutputs != 1 ||
        mB <= 0 || mD <= 0 || mH <= 0 || mW <= 0 ||
        inputs[0] == nullptr || inputs[1] == nullptr || inputs[2] == nullptr ||
        inputs[3] == nullptr || outputs[0] == nullptr) {
        return false;
    }
    if (inputDesc[0].dims.nbDims != 2 || inputDesc[1].dims.nbDims != 2 ||
        inputDesc[2].dims.nbDims != 1 || inputDesc[3].dims.nbDims != 1 ||
        outputDesc[0].dims.nbDims != 5) {
        return false;
    }
    const int64_t n = inputDesc[0].dims.d[0];
    const int64_t c = inputDesc[0].dims.d[1];
    const int64_t geom_n = inputDesc[1].dims.d[0];
    const int64_t geom_width = inputDesc[1].dims.d[1];
    const int64_t starts_n = inputDesc[2].dims.d[0];
    const int64_t lengths_n = inputDesc[3].dims.d[0];
    if (n < 0 || c <= 0 || geom_n != n || geom_width != 4 ||
        starts_n < 0 || lengths_n != starts_n) {
        return false;
    }
    if (outputDesc[0].dims.d[0] != mB || outputDesc[0].dims.d[1] != mD ||
        outputDesc[0].dims.d[2] != mH || outputDesc[0].dims.d[3] != mW ||
        outputDesc[0].dims.d[4] != c) {
        return false;
    }
    if (n > std::numeric_limits<int>::max() || c > std::numeric_limits<int>::max() ||
        starts_n > std::numeric_limits<int>::max()) {
        return false;
    }
    if (inputDesc[0].type != nvinfer1::DataType::kFLOAT ||
        inputDesc[1].type != nvinfer1::DataType::kINT32 ||
        inputDesc[2].type != nvinfer1::DataType::kINT32 ||
        inputDesc[3].type != nvinfer1::DataType::kINT32 ||
        outputDesc[0].type != nvinfer1::DataType::kFLOAT) {
        return false;
    }
    return true;
}

// Output dimensions: [B, D, H, W, C] where C comes from input x
nvinfer1::DimsExprs BEVPoolV2Plugin::getOutputDimensions(
    int32_t outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int32_t nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
    nvinfer1::DimsExprs output{};
    if (outputIndex != 0 || inputs == nullptr || nbInputs != 4 ||
        inputs[0].nbDims != 2 || inputs[1].nbDims != 2 ||
        inputs[2].nbDims != 1 || inputs[3].nbDims != 1 ||
        mB <= 0 || mD <= 0 || mH <= 0 || mW <= 0) {
        output.nbDims = 0;
        return output;
    }

    // inputs[0] = x: [N, C] - flattened features
    // inputs[1] = geom_feats: [N, 4] - coordinates
    // inputs[2] = interval_starts: [M]
    // inputs[3] = interval_lengths: [M]
    
    output.nbDims = 5;
    output.d[0] = exprBuilder.constant(mB);  // B
    output.d[1] = exprBuilder.constant(mD);  // D (Z)
    output.d[2] = exprBuilder.constant(mH);  // H (Y)
    output.d[3] = exprBuilder.constant(mW);  // W (X)
    output.d[4] = inputs[0].d[1];            // C from x
    
    return output;
}

bool BEVPoolV2Plugin::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept {
    if (inOut == nullptr || nbInputs != 4 || nbOutputs != 1 ||
        pos < 0 || pos >= nbInputs + nbOutputs) {
        return false;
    }

    // Support FP32 for all inputs/outputs
    // pos 0: x [N, C], pos 1: geom_feats [N, 4], pos 2: interval_starts [M], pos 3: interval_lengths [M]
    // pos 4: output [B, D, H, W, C]
    if (pos < nbInputs) {
        // Inputs: x is FP32, others are INT32
        if (pos == 0) {
            return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
                   inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
        } else {
            return inOut[pos].type == nvinfer1::DataType::kINT32 &&
                   inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
        }
    } else {
        // Output
        return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
               inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    }
}

void BEVPoolV2Plugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int32_t nbOutputs) noexcept {
    // No dynamic configuration needed
}

size_t BEVPoolV2Plugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int32_t nbOutputs) const noexcept {
    return 0;  // No workspace needed
}

int32_t BEVPoolV2Plugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
    constexpr int32_t kInputCount = 4;
    constexpr int32_t kOutputCount = 1;
    if (!validate_input_contract(inputDesc, kInputCount, outputDesc, kOutputCount, inputs, outputs)) {
        return -1;
    }

    const auto& x_dims = inputDesc[0].dims;           // [N, C]
    const auto& interval_dims = inputDesc[3].dims;    // [M]
    
    int N = x_dims.d[0];
    int C = x_dims.d[1];
    int n_intervals = interval_dims.d[0];
    
    const float* x = static_cast<const float*>(inputs[0]);
    const int* geom_feats = static_cast<const int*>(inputs[1]);
    const int* interval_starts = static_cast<const int*>(inputs[2]);
    const int* interval_lengths = static_cast<const int*>(inputs[3]);
    float* out = static_cast<float*>(outputs[0]);

    const size_t output_elements = static_cast<size_t>(mB) * mD * mH * mW * C;
    if (cudaMemsetAsync(out, 0, output_elements * sizeof(float), stream) != cudaSuccess) {
        return -1;
    }
    
    launch_bev_pool_v2(
        mB, mD, mH, mW, N, n_intervals, C,
        x, geom_feats, interval_starts, interval_lengths,
        out, stream
    );

    // kernel launch 失败（非法配置/资源不足）在这里捕获；异步执行期错误在后续同步点暴露
    if (cudaGetLastError() != cudaSuccess) {
        return -1;
    }

    return 0;
}

nvinfer1::DataType BEVPoolV2Plugin::getOutputDataType(
    int32_t index,
    const nvinfer1::DataType* inputTypes,
    int32_t nbInputs) const noexcept {
    return nvinfer1::DataType::kFLOAT;
}

const char* BEVPoolV2Plugin::getPluginType() const noexcept {
    return BEV_POOL_PLUGIN_NAME;
}

const char* BEVPoolV2Plugin::getPluginVersion() const noexcept {
    return BEV_POOL_PLUGIN_VERSION;
}

int32_t BEVPoolV2Plugin::getNbOutputs() const noexcept {
    return 1;
}

int32_t BEVPoolV2Plugin::initialize() noexcept {
    return 0;
}

void BEVPoolV2Plugin::terminate() noexcept {}

size_t BEVPoolV2Plugin::getSerializationSize() const noexcept {
    return 4 * sizeof(int);  // B, D, H, W
}

void BEVPoolV2Plugin::serialize(void* buffer) const noexcept {
    char* buf = static_cast<char*>(buffer);
    memcpy(buf, &mB, sizeof(int)); buf += sizeof(int);
    memcpy(buf, &mD, sizeof(int)); buf += sizeof(int);
    memcpy(buf, &mH, sizeof(int)); buf += sizeof(int);
    memcpy(buf, &mW, sizeof(int));
}

void BEVPoolV2Plugin::destroy() noexcept {
    delete this;
}

void BEVPoolV2Plugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const char* BEVPoolV2Plugin::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

nvinfer1::IPluginV2DynamicExt* BEVPoolV2Plugin::clone() const noexcept {
    return new BEVPoolV2Plugin(mB, mD, mH, mW);
}

// ==================== Plugin Creator ====================

BEVPoolV2PluginCreator::BEVPoolV2PluginCreator() {
    mFields = {
        {"B", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
        {"D", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
        {"H", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
        {"W", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
    };
    mFC.nbFields = static_cast<int32_t>(mFields.size());
    mFC.fields = mFields.data();
}

const char* BEVPoolV2PluginCreator::getPluginName() const noexcept {
    return BEV_POOL_PLUGIN_NAME;
}

const char* BEVPoolV2PluginCreator::getPluginVersion() const noexcept {
    return BEV_POOL_PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* BEVPoolV2PluginCreator::getFieldNames() noexcept {
    return &mFC;
}

nvinfer1::IPluginV2* BEVPoolV2PluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc) noexcept {
    
    // B/D/H/W 均为必需的输出 shape 参数；任一缺失即视为配置错误，拒绝创建
    int B = 0, D = 0, H = 0, W = 0;
    bool has_B = false, has_D = false, has_H = false, has_W = false;
    if (fc == nullptr || fc->nbFields < 0 ||
        (fc->nbFields > 0 && fc->fields == nullptr)) {
        return nullptr;
    }

    for (int i = 0; i < fc->nbFields; ++i) {
        const auto& f = fc->fields[i];
        if (f.name == nullptr || f.data == nullptr || f.length != 1 ||
            f.type != nvinfer1::PluginFieldType::kINT32) {
            return nullptr;
        }
        if (std::string(f.name) == "B") {
            B = *static_cast<const int*>(f.data); has_B = true;
        } else if (std::string(f.name) == "D") {
            D = *static_cast<const int*>(f.data); has_D = true;
        } else if (std::string(f.name) == "H") {
            H = *static_cast<const int*>(f.data); has_H = true;
        } else if (std::string(f.name) == "W") {
            W = *static_cast<const int*>(f.data); has_W = true;
        } else {
            // 未识别的字段名：拒绝，避免拼写错误被静默忽略
            return nullptr;
        }
    }
    if (!has_B || !has_D || !has_H || !has_W) {
        return nullptr;
    }
    if (B <= 0 || D <= 0 || H <= 0 || W <= 0) {
        return nullptr;
    }
    return new BEVPoolV2Plugin(B, D, H, W);
}

nvinfer1::IPluginV2* BEVPoolV2PluginCreator::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength) noexcept {
    constexpr size_t kSerializationSize = 4 * sizeof(int);
    if (serialData == nullptr || serialLength < kSerializationSize) {
        return nullptr;
    }
    return new BEVPoolV2Plugin(serialData, serialLength);
}

void BEVPoolV2PluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const char* BEVPoolV2PluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

// Explicit initialization function
extern "C" {
    __attribute__((visibility("default")))
    void forceInitBEVPoolV2Plugin() {
        // Empty: actual registration done by REGISTER_TENSORRT_PLUGIN
    }
}

// Auto-registration
REGISTER_TENSORRT_PLUGIN(BEVPoolV2PluginCreator);
