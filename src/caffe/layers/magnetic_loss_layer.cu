#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/magnetic_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void compute_Xi_Cj_dist(int nthreads, const int C,const int D,
	      const Dtype* bottom,
	      const Dtype* center, 
		  Dtype* diff, Dtype* dist) 
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    int m = index / C;
    int c = index % C;
    // diff(i) = Xi - Cj
	// dist[index] = Sum{d^k}
	Dtype sum = 0;
	for (int k = 0; k < D; k++) {
		diff[m * C * D + c * D + k] = bottom[m * D + k] - center[c * D + k];
		sum += diff[m * C * D + c * D + k] * diff[m * C * D + c * D + k];
	}
	dist[index] = sum;
  }
}

template <typename Dtype>
__global__ void kernel_set_multiplier(const int M, const int C, Dtype val,
	const Dtype* Y, Dtype* Z) {
	CUDA_KERNEL_LOOP(index, M) {
		int Yi = Y[index];
		Z[index * C + Yi] = val;
	}
}

template <typename Dtype>
__global__ void kernel_add_margin(const int M, const int C, Dtype val,
	const Dtype* Y, Dtype* Z) {
	CUDA_KERNEL_LOOP(index, M) {
		int Yi = Y[index];
		Z[index * C + Yi] -= val;
	}
}

template <typename Dtype>
__global__ void kernel_set_backward_multiplier(const int M, const int C, Dtype lambda,
	const Dtype* prob_data, const Dtype* Y, Dtype* mutiplier_data) {
	CUDA_KERNEL_LOOP(index, M) {
		int Yi = Y[index];
		Dtype* scale = mutiplier_data + index * C;
		scale[Yi] = lambda * (prob_data[index * C + Yi] - 1);
	}
}

template <typename Dtype>
__global__ void kernel_channel_max(const int M, const int C,
	const Dtype* data, Dtype* out) {
	CUDA_KERNEL_LOOP(index, M) {
		Dtype maxval = -FLT_MAX;
		for (int c = 0; c < C; ++c) {
			maxval = max(data[index * C + c], maxval);
		}
		out[index] = maxval;
	}
}

template <typename Dtype>
__global__ void kernel_channel_subtract(int nthreads,const int M, const int C,
	const Dtype* channel_max, Dtype* data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int i = index / C;
		int j = index % C;
		data[index] -= channel_max[i];
	}
}

template <typename Dtype>
__global__ void kernel_exp(const int count, const Dtype* data, Dtype* out) {
	CUDA_KERNEL_LOOP(index, count) {
		out[index] = exp(data[index]);
	}
}

template <typename Dtype>
__global__ void kernel_channel_sum(const int M, const int C,
	const Dtype* data, Dtype* out) {
	CUDA_KERNEL_LOOP(index, M) {
		Dtype sum = 0;
		for (int c = 0; c < C; ++c) {
			sum += data[index * C + c];
		}
		out[index] = sum;
	}
}

template <typename Dtype>
__global__ void kernel_channel_div(int nthreads, const int M, const int C,
	const Dtype* channel_sum, Dtype* data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int i = index / C;
		int j = index % C;
		data[index] /= channel_sum[i];
	}
}

template <typename Dtype>
__global__ void compute_loss(const int nthreads,const int C,
	const Dtype* prob_data, const Dtype* label,Dtype* loss) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int label_value = static_cast<int>(label[index]);
		loss[index] = -log(max(prob_data[index * C +label_value], Dtype(FLT_MIN)));
	}
}

template <typename Dtype>
__global__ void kernel_channel_dot(const int num, const int channels,
	const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
	Dtype* channel_dot) {
	CUDA_KERNEL_LOOP(index, num * spatial_dim) {
		int n = index / spatial_dim;
		int s = index % spatial_dim;
		Dtype dot = 0;
		for (int c = 0; c < channels; ++c) {
			dot += (data_1[(n * channels + c) * spatial_dim + s]
				* data_2[(n * channels + c) * spatial_dim + s]);
		}
		channel_dot[index] = dot;
	}
}

template <typename Dtype>
__global__ void compute_center_diff(const int M, const int C, const int D,
        const Dtype* scale_data, const Dtype* diff_data,
        Dtype* center_diff) {
  CUDA_KERNEL_LOOP(index, C*D) {
	int c = index / D;
	int d = index % D;
	Dtype sum = 0;
    for (int m = 0; m < M; m++) {
		sum += diff_data[m * C* D + c * D + d] * scale_data[m * C + c];
    }
	center_diff[index] = sum;
  }
}

template <typename Dtype>
__global__ void compute_bottom_diff(const int M, const int C, const int D,
	const Dtype* scale_data, const Dtype* diff_data,
	Dtype* bottom_diff) {
	CUDA_KERNEL_LOOP(index, M*D) {
		int m = index / D;
		int d = index % D;
		Dtype sum = 0;
		for (int c = 0; c < C; c++) {
			sum += diff_data[m * C* D + c * D + d] * scale_data[m * C + c];
		}
		bottom_diff[m * D + d] = sum;
	}
}

template <typename Dtype>
void printData(const Dtype* data,int rows,int cols)
{
	printf("\n");
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			printf("%f ", data[i*cols + j]);
		}
		printf("\n");
	}
}

template <typename Dtype>
void printBlob(const char* tag, Blob<Dtype>* blob)
{
	printf("%s", tag);
	printData(blob->cpu_data(), blob->num(), blob->count() / blob->num());
}

template <typename Dtype>
Dtype* SyncBlobCPU(Blob<Dtype>* blob)
{
	Dtype* cpu = blob->mutable_cpu_data();
	Dtype* gpu = blob->mutable_cpu_data();
	caffe_gpu_memcpy(blob->data()->size(), gpu, cpu);
	return cpu;
}

template <typename Dtype>
void MagneticLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int nthreads = M * C;
  const Dtype* X = bottom[0]->gpu_data();
  const Dtype* Y = bottom[1]->gpu_data();
  const Dtype* center = this->blobs_[0]->gpu_data();
  Dtype* diff_data = diff.mutable_gpu_data();
  Dtype* dist_data = dist.mutable_gpu_data();
  Dtype* scale_data = multiplier.mutable_gpu_data();
  Dtype* prob_data = prob.mutable_gpu_data();
  Dtype* maxD_data = maxD_.mutable_gpu_data();
  Dtype* sumD_data = sumD_.mutable_gpu_data();
  // diff = Xi - Cj 
  // dist = ||diff||^2
  compute_Xi_Cj_dist<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
	  CAFFE_CUDA_NUM_THREADS >> >(nthreads, C, D, X, center, diff_data, dist_data);

  //printBlob("dist", &dist);
  // scale = -1 | -lambda
  caffe_gpu_set<Dtype>(M*C, -1, scale_data);
  if (this->phase_ == TRAIN)
  {
	  kernel_set_multiplier << <CAFFE_GET_BLOCKS(M),
		  CAFFE_CUDA_NUM_THREADS >> >(M, C, -lambda, Y, scale_data);
  }
  // scale x dist -> dist
  caffe_gpu_mul<Dtype>(M*C, scale_data, dist_data, dist_data);
  if (this->phase_ == TRAIN)
  {
	  kernel_add_margin<< <CAFFE_GET_BLOCKS(M),
		  CAFFE_CUDA_NUM_THREADS >> >(M, C, large_margin, Y, scale_data);
  }
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  // compute max
  kernel_channel_max<Dtype> << <CAFFE_GET_BLOCKS(M),
	  CAFFE_CUDA_NUM_THREADS >> >(M, C, dist_data, maxD_data);
  // subtract
  kernel_channel_subtract<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
	  CAFFE_CUDA_NUM_THREADS >> >(nthreads, M, C, maxD_data, dist_data);
  // exponentiate
  kernel_exp<Dtype> << <CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS >> >(
	  nthreads, dist_data, prob_data);
  // sum after exp
  kernel_channel_sum<Dtype> << <CAFFE_GET_BLOCKS(M),
	  CAFFE_CUDA_NUM_THREADS >> >(M, C, prob_data, sumD_data);
  // divide
  kernel_channel_div<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
	  CAFFE_CUDA_NUM_THREADS >> >(nthreads, M, C, sumD_data, prob_data);
  // sumD pYi
  compute_loss<Dtype> << <CAFFE_GET_BLOCKS(M),
	  CAFFE_CUDA_NUM_THREADS >> >(M, C, prob_data, Y, sumD_data);
  //printBlob("prob", &prob);
  //printBlob("loss", &sumD_);
  // sum
  Dtype loss;
  caffe_gpu_asum(M, sumD_data, &loss);
  loss = loss / M;
  //LOG(INFO) << "Loss:" << loss;
  top[0]->mutable_cpu_data()[0] = loss;
  if (top.size() == 2) {
	  top[1]->ShareData(prob);
  }
}

template <typename Dtype>
void MagneticLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

	const Dtype* label = bottom[1]->gpu_data();
	Dtype* scale_data = multiplier.mutable_gpu_data();
	Dtype* prob_data = prob.mutable_gpu_data();
	
	// set multiplier
	//caffe_set(M*C, (Dtype)0., scale_data);
	//caffe_sub(M*C, scale_data, prob_data, scale_data);
	caffe_copy(M*C, prob_data, scale_data);
	kernel_set_backward_multiplier << <CAFFE_GET_BLOCKS(M),
		CAFFE_CUDA_NUM_THREADS >> >(M, C, lambda, prob_data, label, scale_data);
	//printBlob("multiplier", &multiplier);
	// Gradient with respect to centers
	if (update_center && this->param_propagate_down_[0])
	{
		Dtype* center_diff = this->blobs_[0]->mutable_gpu_diff();
		const Dtype* diff_data = diff.gpu_data();
		// Zero
		caffe_gpu_set(C * D, (Dtype)0., center_diff);
		compute_center_diff << <CAFFE_GET_BLOCKS(C*D),
			CAFFE_CUDA_NUM_THREADS >> >(M, C, D, scale_data, diff_data, center_diff);
		// x 1/M
		caffe_gpu_scal(C * D, (Dtype)1. / (Dtype)M, center_diff);
	}
	// Gradient with respect to bottom data 
	if (propagate_down[0])
	{
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* label = bottom[1]->gpu_data();

		Dtype* diff_data = diff.mutable_gpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		// Zero
		caffe_gpu_set(M * D, (Dtype)0., bottom_diff);
		compute_bottom_diff << <CAFFE_GET_BLOCKS(M*D),
			CAFFE_CUDA_NUM_THREADS >> >(M, C, D, scale_data, diff_data, bottom_diff);
		// x 1/M
		caffe_gpu_scal(M * D, -top[0]->cpu_diff()[0] / M, bottom_diff);
	}
	if (propagate_down[1]) {
		LOG(FATAL) << this->type()
			<< " Layer cannot backpropagate to label inputs.";
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(MagneticLossLayer);

}  // namespace caffe
