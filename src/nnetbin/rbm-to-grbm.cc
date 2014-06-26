/*
 * nnetbin/rbm-to-grbm.cc
 *
 * Convert the original Kaldi unit variance Guassian-Bernoulli RBM to variable
 * variance GRBM.
 *
 *  Created on: May 13, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet/nnet-component.h"

int main(int argc, char *argv[]) {

  using namespace kaldi;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
        "Convert conventional unit variance GRBM to variable GRBM.\n"
            "Usage:  rbm-to-grbm [options] <model-in> <model-out>\n"
            "e.g.: \n"
            " rbm-to-grbm rbm.mdl grbm.mdl\n";

    ParseOptions po(usage);
    bool binary = false;
    po.Register("binary", &binary, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string input_filename = po.GetArg(1), target_filename = po.GetArg(2);

    bool input_binary;
    Input in(input_filename, &input_binary);

    int32 dim_out, dim_in;
    std::string token;
    std::string vis_node_type, hid_node_type;
    Matrix<BaseFloat> vis_hid;
    Vector<BaseFloat> vis_bias, hid_bias, vis_std;

    int first_char = Peek(in.Stream(), input_binary);
    if (first_char == EOF)
      return -1;

    /* read layer type */
    ReadToken(in.Stream(), input_binary, &token);
    Component::ComponentType comp_type = Component::MarkerToType(token);
    KALDI_ASSERT(comp_type == Component::kRbm);

    /* read layer dimensions */
    ReadBasicType(in.Stream(), input_binary, &dim_out);
    ReadBasicType(in.Stream(), input_binary, &dim_in);

    /* read layer types */
    ReadToken(in.Stream(), input_binary, &vis_node_type);
    ReadToken(in.Stream(), input_binary, &hid_node_type);

    KALDI_ASSERT(vis_node_type == "gauss");
    KALDI_ASSERT(hid_node_type == "bern");

    /* read layer weights */
    vis_hid.Read(in.Stream(), input_binary);
    vis_bias.Read(in.Stream(), input_binary);
    hid_bias.Read(in.Stream(), input_binary);

    KALDI_ASSERT(vis_hid.NumRows() == dim_out && vis_hid.NumCols() == dim_in);
    KALDI_ASSERT(vis_bias.Dim() == dim_in);
    KALDI_ASSERT(hid_bias.Dim() == dim_out);

    in.Close();
    /*======== READ Section done! =========*/

    /* set the variance to 1 */
    vis_std.Resize(dim_in, kSetZero);
    vis_std.Set(1.0);

    /* open target file */
    Output out(target_filename, binary, true);

    WriteToken(out.Stream(), binary, Component::TypeToMarker(Component::kGRbm));
    WriteBasicType(out.Stream(), binary, dim_out);
    WriteBasicType(out.Stream(), binary, dim_in);
    if (!binary)
      out.Stream() << "\n";

    WriteToken(out.Stream(), binary, "gauss");
    WriteToken(out.Stream(), binary, "bern");

    vis_hid.Write(out.Stream(), binary);
    vis_bias.Write(out.Stream(), binary);
    hid_bias.Write(out.Stream(), binary);
    vis_std.Write(out.Stream(), binary);

    out.Close();
    /*======== Write Section done! =======*/

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

