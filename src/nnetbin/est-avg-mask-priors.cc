// nnetbin/est-avg-mask-priors.cc
//
// Compute the pdf dependent mask priors, which are simply the
// average mask pattern among all the training data.
//

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/timer.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
        "Estimate the Average Mask Priors for each PDFs.\n"
            "Usage:  est-avg-mask-priors [options] <mask-rspecifier> <pdf-rspecifier> <out-wxfilename> [<pdfcounts-wxfilename>]\n"
            "e.g.: \n"
            " est-avg-mask-priors --binary=false --num-pdfs=120 --dim=129 scp:mask.scp ark:pdf.scp mask_patterns\n";

    ParseOptions po(usage);
    bool binary = true;
    po.Register("binary", &binary, "Write output patterns in binary mode");
    int32 num_pdfs = 0;
    po.Register("num-pdfs", &num_pdfs, "Number of patterns to compute");
    int32 dim = 0;
    po.Register("dim", &dim, "Dimension of the masks");
    po.Read(argc, argv);

    if (po.NumArgs() != 3 && po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string mask_rspecifier = po.GetArg(1),
        pdf_rspecifier = po.GetArg(2),
        out_wxfilename = po.GetArg(3),
        pdfcounts_wxfilename = po.GetOptArg(4);

    if (num_pdfs == 0 || dim == 0) {
      KALDI_ERR<< "Incorrect configuration parameters for --num-pdfs or --dim";
    }

    SequentialBaseFloatMatrixReader mask_reader(mask_rspecifier);
    RandomAccessInt32VectorReader pdf_reader(pdf_rspecifier);

    Timer tim;

    int32 num_done = 0, num_no_tgt_pdf = 0, num_other_error = 0;
    Matrix<BaseFloat> patterns(num_pdfs, dim, kSetZero);
    Vector<BaseFloat> counts(num_pdfs, kSetZero);

    while (!mask_reader.Done()) {
      // get the keys
      std::string utt = mask_reader.Key();
      const Matrix<BaseFloat> &mask = mask_reader.Value();

      if (!pdf_reader.HasKey(utt)) {
        KALDI_WARN<< "No PDF labels fro: " << utt << ", utterance ignored!";
        ++num_no_tgt_pdf;
        continue;
      }

      std::vector<int32> labs = pdf_reader.Value(utt);

      if (mask.NumRows() != labs.size()) {
        KALDI_WARN<< "Label has wrong size " << (labs.size()) << " vs. " << (mask.NumRows());
        ++num_other_error;
        continue;
      }

      for (int32 i = 0; i < mask.NumRows(); ++i) {
        (patterns.Row(labs[i])).AddVec(1.0, mask.Row(i));
        counts(labs[i])++;
        }

      num_done++;
      mask_reader.Next();

      if(num_done % 1000 == 0){
        KALDI_LOG << "Done " << num_done << " files.";
      }

    }

    for (int32 i = 0; i < num_pdfs; ++i) {
      (patterns.Row(i)).Scale(1.0 / counts(i));
    }

    // write the patterns
    {
      Output ko(out_wxfilename, binary);
      patterns.Write(ko.Stream(), binary);
    }

    // write the counts if required
    if (pdfcounts_wxfilename != "") {
      Output ko(pdfcounts_wxfilename, binary);
      counts.Write(ko.Stream(), binary);
    }

    std::cout << "\n" << std::flush;
    KALDI_LOG<< "COMPUTATION" << " FINISHED " << tim.Elapsed() << "s";
    KALDI_LOG<< "Done " << num_done << " files, " << num_no_tgt_pdf << " with no tgt_pdfs, " << num_other_error << " with other errors.";

    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
