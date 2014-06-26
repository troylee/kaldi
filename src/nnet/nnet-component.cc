// nnet/nnet-component.cc

// Copyright 2011  Karel Vesely

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "nnet/nnet-component.h"

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-activation.h"
#include "nnet/nnet-biasedlinearity.h"
#include "nnet/nnet-rbm.h"
#include "nnet/nnet-various.h"
#include "nnet/nnet-dropoutbl.h"
#include "nnet/nnet-cmvnbl.h"
#include "nnet/nnet-posnegbl.h"
#include "nnet/nnet-gaussbl.h"
#include "nnet/nnet-maskedbl.h"
#include "nnet/nnet-rorbm.h"
#include "nnet/nnet-grbm.h"
#include "nnet/nnet-linbl.h"
#include "nnet/nnet-linrbm.h"
#include "nnet/nnet-hmmbl.h"
#include "nnet/nnet-codebl.h"

namespace kaldi {

const struct Component::key_value Component::kMarkerMap[] = { {
    Component::kBiasedLinearity, "<biasedlinearity>" }, { Component::kSigmoid,
    "<sigmoid>" }, { Component::kSoftmax, "<softmax>" }, { Component::kRbm,
    "<rbm>" }, { Component::kExpand, "<expand>" },
    { Component::kCopy, "<copy>" }, { Component::kDropoutBL, "<dropoutbl>" },
    { Component::kRelu, "<relu>"}, {Component::kSoftRelu, "<softrelu>"},
    {Component::kCMVNBL, "<cmvnbl>"}, {Component::kPosNegBL, "<posnegbl>"},
    {Component::kGaussBL, "<gaussbl>"}, {Component::kMaskedBL, "<maskedbl>"},
    {Component::kMaskedRbm, "<maskedrbm>"}, {Component::kRoRbm, "<rorbm>"},
    {Component::kGRbm, "<grbm>"}, {Component::kLinBL, "<linbl>"},
    {Component::kLinRbm, "<linrbm>"}, {Component::kHMMBL, "<hmmbl>"},
    {Component::kCodeBL, "<codebl>"}};

const char* Component::TypeToMarker(ComponentType t) {
  int32 N = sizeof(kMarkerMap) / sizeof(kMarkerMap[0]);
  for (int i = 0; i < N; i++) {
    if (kMarkerMap[i].key == t)
      return kMarkerMap[i].value;
  }
  KALDI_ERR<< "Unknown type" << t;
  return NULL;
}

Component::ComponentType Component::MarkerToType(const std::string &s) {
  int32 N = sizeof(kMarkerMap) / sizeof(kMarkerMap[0]);
  for (int i = 0; i < N; i++) {
    if (0 == strcmp(kMarkerMap[i].value, s.c_str()))
      return kMarkerMap[i].key;
  }
  KALDI_ERR<< "Unknown marker" << s;
  return kUnknown;
}

Component* Component::Read(std::istream &is, bool binary, Nnet *nnet) {
  int32 dim_out, dim_in;
  std::string token;

  int first_char = Peek(is, binary);
  if (first_char == EOF)
    return NULL;

  ReadToken(is, binary, &token);
  Component::ComponentType comp_type = Component::MarkerToType(token);

  ReadBasicType(is, binary, &dim_out);
  ReadBasicType(is, binary, &dim_in);

  Component *p_comp = NULL;
  switch (comp_type) {
    case Component::kBiasedLinearity:
      p_comp = new BiasedLinearity(dim_in, dim_out, nnet);
      break;
    case Component::kSigmoid:
      p_comp = new Sigmoid(dim_in, dim_out, nnet);
      break;
    case Component::kSoftmax:
      p_comp = new Softmax(dim_in, dim_out, nnet);
      break;
    case Component::kRbm:
      p_comp = new Rbm(dim_in, dim_out, nnet);
      break;
    case Component::kMaskedRbm:
      p_comp = new MaskedRbm(dim_in, dim_out, nnet);
      break;
    case Component::kRoRbm:
      p_comp = new RoRbm(dim_in, dim_out, nnet);
      break;
    case Component::kGRbm:
      p_comp = new GRbm(dim_in, dim_out, nnet);
      break;
    case Component::kLinRbm:
      p_comp = new LinRbm(dim_in, dim_out, nnet);
      break;
    case Component::kExpand:
      p_comp = new Expand(dim_in, dim_out, nnet);
      break;
    case Component::kCopy:
      p_comp = new Copy(dim_in, dim_out, nnet);
      break;
    case Component::kDropoutBL:
      p_comp = new DropoutBL(dim_in, dim_out, nnet);
      break;
    case Component::kCMVNBL:
      p_comp = new CMVNBL(dim_in, dim_out, nnet);
      break;
    case Component::kPosNegBL:
      p_comp = new PosNegBL(dim_in, dim_out, nnet);
      break;
    case Component::kGaussBL:
      p_comp = new GaussBL(dim_in, dim_out, nnet);
      break;
    case Component::kMaskedBL:
      p_comp = new MaskedBL(dim_in, dim_out, nnet);
      break;
    case Component::kLinBL:
      p_comp = new LinBL(dim_in, dim_out, nnet);
      break;
    case Component::kHMMBL:
      p_comp = new HMMBL(dim_in, dim_out, nnet);
      break;
    case Component::kCodeBL:
      p_comp = new CodeBL(dim_in, dim_out, nnet);
      break;
    case Component::kRelu:
      p_comp = new Relu(dim_in, dim_out, nnet);
      break;
    case Component::kSoftRelu:
      p_comp = new SoftRelu(dim_in, dim_out, nnet);
      break;
    case Component::kUnknown:
    default:
      KALDI_ERR<< "Missing type: " << token;
    }

  p_comp->ReadData(is, binary);
  return p_comp;
}

void Component::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, Component::TypeToMarker(GetType()));
  WriteBasicType(os, binary, OutputDim());
  WriteBasicType(os, binary, InputDim());
  if (!binary)
    os << "\n";
  this->WriteData(os, binary);
}

void Component::WriteAsBiasedLinearity(std::ostream &os, bool binary) const {
  KALDI_ASSERT(GetType()==Component::kDropoutBL);
}

}  // namespace
