// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

typedef OrtCustomOp const* CPTR_OrtCustomOp;
typedef CPTR_OrtCustomOp (*FxGetSchemaInstance)();

FxGetSchemaInstance const * GetCustomOpSchemaList();

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi& ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};
