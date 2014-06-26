/*
 * rorbm-init.cc
 *
 *  Create the initial RoRbm model from a clean GRBM.
 *
 *  Created on: May 15, 2013
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
        "Create an initial RoRbm model from a clean GRBM.\n"
            "Usage:  rorbm-init [options] <clean-model-in> <model-out>\n"
            "e.g.: \n"
            " rorbm-init clean_grbm.mdl rorbm.mdl\n";

    ParseOptions po(usage);
    bool binary = false;
    po.Register("binary", &binary, "Write output in binary mode");

    int32 hid_dim = 1024;
    po.Register("hid-dim", &hid_dim, "Hidden layer dimensionality of the noise RBM model.");

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
    KALDI_ASSERT(comp_type == Component::kGRbm);

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
    vis_std.Read(in.Stream(), input_binary);

    KALDI_ASSERT(vis_hid.NumRows() == dim_out && vis_hid.NumCols() == dim_in);
    KALDI_ASSERT(vis_bias.Dim() == dim_in);
    KALDI_ASSERT(hid_bias.Dim() == dim_out);
    KALDI_ASSERT(vis_std.Dim() == dim_in);

    in.Close();
    /*======== READ Section done! =========*/

    /* Initialize the RoRbm parameters */
    CuMatrix<BaseFloat> U(hid_dim, dim_in);
    CuVector<BaseFloat> d(dim_in);
    CuVector<BaseFloat> e(hid_dim);
    CuVector<BaseFloat> bt(dim_in), lamt2(dim_in), gamma2(dim_in);

    U.SetZero();
    d.Set(3.0);
    e.SetZero();
    bt.SetZero();
    gamma2.Set(50.0);
    lamt2.Set(1.0);

    /* open target file */
    Output out(target_filename, binary, true);

    WriteToken(out.Stream(), binary, Component::TypeToMarker(Component::kRoRbm));
    WriteBasicType(out.Stream(), binary, dim_out);
    WriteBasicType(out.Stream(), binary, dim_in);
    if (!binary)
      out.Stream() << "\n";

    // write layer types
    WriteToken(out.Stream(), binary, "gauss");
    WriteToken(out.Stream(), binary, "bern");
    WriteToken(out.Stream(), binary, "bern");

    // write noise hidden dim
    WriteBasicType(out.Stream(), binary, hid_dim);

    // write clean GRBM
    vis_hid.Write(out.Stream(), binary);
    vis_bias.Write(out.Stream(), binary);
    hid_bias.Write(out.Stream(), binary);
    vis_std.Write(out.Stream(), binary);

    // write noise RBM
    U.Write(out.Stream(), binary);
    d.Write(out.Stream(), binary);
    e.Write(out.Stream(), binary);

    // write visible 3-way factorization parameters
    bt.Write(out.Stream(), binary);
    gamma2.Write(out.Stream(), binary);
    lamt2.Write(out.Stream(), binary);

    out.Close();
    /*======== Write Section done! =======*/

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}




