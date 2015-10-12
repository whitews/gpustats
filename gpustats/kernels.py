from gpustats.codegen import MVDensityKernel, Exp, CUFile

# TODO: check for name conflicts!

_log_pdf_mvnormal = """
__device__ float %(name)s(float* data, float* params, int dim) {
  unsigned int LOGDET_OFFSET = dim * (dim + 3) / 2;
  float* mean = params;
  float* sigma = params + dim;
  float mult = params[LOGDET_OFFSET];
  float logdet = params[LOGDET_OFFSET + 1];

  float discrim = 0;
  float sum;
  unsigned int i, j;
  for (i = 0; i < dim; ++i)
  {
    sum = 0;
    for(j = 0; j <= i; ++j) {
      sum += *sigma++ * (data[j] - mean[j]);
    }
    discrim += sum * sum;
  }
  return log(mult) - 0.5f * (discrim + logdet + LOG_2_PI * dim);
}
"""

log_pdf_mvnormal = MVDensityKernel('log_pdf_mvnormal', _log_pdf_mvnormal)
pdf_mvnormal = Exp('pdf_mvnormal', log_pdf_mvnormal)
sample_discrete = CUFile('sample_discrete', 'sampleFromMeasureMedium.cu')
