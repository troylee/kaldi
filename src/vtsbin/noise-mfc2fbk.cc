/*
 * noise-mfc2fbk.cc
 *
 *  Created on: Jan 15, 2013
 *      Author: Troy Lee (troy.lee2008@gmail.com)
 *
 *  Convert MFCC noise parameters to FBank parameters.
 *
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

#include "vts/vts-first-order.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Convert noise parameters from parameter kind MFCC to FBank\n"
        "Usage: noise-mfc2fbk [options] noise-mfcc-rspecifier noise-fbank-wspecifier\n";

    ParseOptions po(usage);

    int32 delta_order = 2; // 0-static, 1-delta, 2-accelerate
    int32 num_cepstral = 13;
    int32 num_fbank = 26;
    BaseFloat ceplifter = 22;

    po.Register("delta-order", &delta_order,
                "The feature's delta order: [0-static, 1-delta, 2-accelerate]");
    po.Register("num-cepstral", &num_cepstral, "Number of Cepstral features");
    po.Register("num-fbank", &num_fbank,
                "Number of FBanks used to generate the Cepstral features");
    po.Register("ceplifter", &ceplifter,
                "CepLifter value used for feature extraction");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string mfcc_rspecifier = po.GetArg(1);
    std::string fbank_wspecifier = po.GetArg(2);

    SequentialDoubleVectorReader mfcc_reader(mfcc_rspecifier);
    DoubleVectorWriter fbank_writer(fbank_wspecifier);

    Matrix<double> dct_mat, inv_dct_mat;
    GenerateDCTmatrix(num_cepstral, num_fbank, ceplifter, &dct_mat,
                      &inv_dct_mat);

    // compute the FBank noise estimate
    Vector<double> mu(num_fbank * (delta_order+1), kSetZero);
    Vector<double> var(num_fbank * (delta_order+1), kSetZero);
    Matrix<double> tmp_mfcc(num_cepstral, num_cepstral), tmp_fbank(
                num_fbank, num_fbank);

    for (; !mfcc_reader.Done(); mfcc_reader.Next()) {
      std::string key = mfcc_reader.Key();
      Vector<double> feat(mfcc_reader.Value());

      size_t pos = key.rfind("_var_z");
      if (pos == std::string::npos) {  // mu_h or mu_z
        mu.SetZero();

        SubVector<double> static_mu(mu, 0, num_fbank);
        static_mu.AddMatVec(1.0, inv_dct_mat, kNoTrans,
                            SubVector<double>(feat, 0, num_cepstral),
                            0.0);

        fbank_writer.Write(key, mu);

      } else {  // var
        var.SetZero();

        SubVector<double> static_var(var, 0, num_fbank);
        tmp_mfcc.SetZero();
        tmp_mfcc.CopyDiagFromVec(
            SubVector<double>(feat, 0, num_cepstral));
        tmp_fbank.AddMatMatMat(1.0, inv_dct_mat, kNoTrans, tmp_mfcc, kNoTrans,
                               inv_dct_mat,
                               kTrans, 0.0);
        static_var.CopyDiagFromMat(tmp_fbank);

        if (delta_order >= 1) {
          SubVector<double> delta_var(var, num_fbank, num_fbank);
          tmp_mfcc.SetZero();
          tmp_mfcc.CopyDiagFromVec(
              SubVector<double>(feat, num_cepstral, num_cepstral));
          tmp_fbank.AddMatMatMat(1.0, inv_dct_mat, kNoTrans, tmp_mfcc, kNoTrans,
                                 inv_dct_mat,
                                 kTrans, 0.0);
          delta_var.CopyDiagFromMat(tmp_fbank);

          if (delta_order >= 2) {
            SubVector<double> acc_var(var, num_fbank << 1, num_fbank);
            tmp_mfcc.SetZero();
            tmp_mfcc.CopyDiagFromVec(
                SubVector<double>(feat, num_cepstral << 1, num_cepstral));
            tmp_fbank.AddMatMatMat(1.0, inv_dct_mat, kNoTrans, tmp_mfcc,
                                   kNoTrans,
                                   inv_dct_mat,
                                   kTrans, 0.0);
            acc_var.CopyDiagFromMat(tmp_fbank);

            if(delta_order > 2) {
              KALDI_ERR << "Delta order higher than 2 is not supported yet!";
            }
          }
        }

        fbank_writer.Write(key, var);

      }

    }
    return 0;

  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

