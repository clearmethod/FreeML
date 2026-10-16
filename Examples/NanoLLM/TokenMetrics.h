#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace NanoLLM {

// DataRead already accounts for a matrix's storage offset. Read each matrix
// once, then index samples directly instead of allocating GPU view metadata.
template<class Mat>
std::string DecodeTokenBatch(Mat* matrix, const std::vector<char>& vocabulary, uint32_t batch=0u)
{
    if (!matrix) return {};
    const auto& shape=matrix->GetShape();
    if (shape.channels!=1u || !shape.width || batch>=shape.batch)
        throw std::invalid_argument("Token decoding requires a valid sequence batch.");
    const auto* data=matrix->DataRead()+uint64_t(batch)*shape.height*shape.width;
    std::string decoded;
    decoded.reserve(shape.height);
    for (uint32_t token=0;token<shape.height;++token) {
        const auto* row=data+uint64_t(token)*shape.width;
        uint32_t index=0u;
        if (shape.width==1u) index=static_cast<uint32_t>(row[0]);
        else for (uint32_t feature=1;feature<shape.width;++feature)
            if (row[feature]>row[index]) index=feature;
        decoded.push_back(index<vocabulary.size()?vocabulary[index]:'?');
    }
    return decoded;
}

template<class Mat>
uint64_t CountCorrectTokens(Mat* logits, Mat* targets)
{
    if (!logits || !targets)
        throw std::invalid_argument("Token accuracy requires logits and targets.");
    const auto& shape=logits->GetShape();
    const auto& labels=targets->GetShape();
    if (shape.channels!=1u || !shape.width || labels.channels!=1u || labels.width!=1u
        || labels.batch!=shape.batch || labels.height!=shape.height)
        throw std::invalid_argument("Token accuracy requires matching sequence batches.");
    const auto* predictions=logits->DataRead();
    const auto* expected=targets->DataRead();
    const uint64_t rows=uint64_t(shape.batch)*shape.height;
    uint64_t correct=0u;
    for (uint64_t token=0;token<rows;++token) {
        const auto* row=predictions+token*shape.width;
        uint32_t best=0u;
        for (uint32_t feature=1;feature<shape.width;++feature)
            if (row[feature]>row[best]) best=feature;
        if (expected[token]==best) ++correct;
    }
    return correct;
}

} // namespace NanoLLM
