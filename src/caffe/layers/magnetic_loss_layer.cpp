#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/magnetic_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype L2Distance(int K, const Dtype* x, const Dtype* y)
{
	Dtype d = 0;
	for (int i = 0; i < K; i++)
	{
		d += (x[i] - y[i]) * (x[i] - y[i]);
	}
	return d;
}

template <typename Dtype>
Dtype L2Norm(int K, Dtype* x, Dtype r)
{
	Dtype d = 0;
	for (int i = 0; i < K; i++)
	{
		d += x[i] * x[i];
	}
	d = sqrt(d);
	if (d < r)
	{
		return d;
	}

	r = r / d;
	for (int i = 0; i < K; i++)
	{
		x[i] *= r ;
	}
	return d;
}

template <typename Dtype>
Dtype ExpandCenterOnce(int N, int K, Dtype* centers, Dtype* dist, float lr, float r)
{
	// find nearest pair
	Dtype minDist = 1e10;
	Dtype sum = 0;
	int count = 0;
	int a, b;
	for (int i = 0; i < N; i++)
	{
		for (int j = i + 1; j < N; j++)
		{
			// <i,j>
			Dtype dot = L2Distance(K, centers + i * K, centers + j * K);
			sum += sqrt(dot);
			count++;
			if (dot < minDist)
			{
				minDist = dot;
				a = i;
				b = j;
			}
		}
	}
	// expand
	Dtype d2 = 0;
	for (int i = 0; i < K; i++)
	{
		dist[i] = centers[a*K + i] - centers[b*K + i];
		d2 += dist[i] * dist[i];
	}
	d2 = sqrt(d2);

	Dtype alpha = lr / d2;
	//printf("pair<%d,%d> dist:%f \n",a,b,minDist);
	// ci += lr/d2 * dist
	for (int i = 0; i < K; i++)
	{
		centers[a*K + i] = centers[a*K + i] + alpha * dist[i];
		centers[b*K + i] = centers[b*K + i] - alpha * dist[i];
	}
	// clip
	Dtype* x = centers + a * K;
	L2Norm<Dtype>(K,x,r);

	x = centers + b * K;
	L2Norm<Dtype>(K, x, r);

	return sqrt(minDist)*count / sum;
}

template <typename Dtype>
void ExpandCenter(int N, int K, Dtype* centers, float lr, int maxIters, float r)
{
	// random init
	LOG(INFO) << "Init centers";
	srand(time(0));
	Dtype* tmp = new Dtype[K];
	LOG(INFO) << "Expand centers";
	for (int i = 0; i < maxIters; i++)
	{
		float stable = ExpandCenterOnce(N, K, centers, tmp, lr, r);
		if (stable > 0.99)
		{
			LOG(INFO) << "Centers are now stable : " << stable << ",iter: " << i;
			break;
		}
		if (i % 1000 == 0)
		{
			LOG(INFO) << "Expand  iter: " << i << " stable:" << stable;
		}
	}
	delete tmp;

	// display centers
	printf("Centers:\n");
	for (int i = 0; i < N; i++)
	{
		printf("c[%d] ", i);
		for (int j = 0; j < K; j++)
		{
			printf("%.3f ", centers[i*K + j]);
		}
		printf("\n");
	}
	return;
}

template <typename Dtype>
void MagneticLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	// LossLayers have a non-zero (1) loss by default.
	if (this->layer_param_.loss_weight_size() == 0) {
		this->layer_param_.add_loss_weight(Dtype(1));
		for (int i = 1; i < top.size(); i++) {
			this->layer_param_.add_loss_weight(Dtype(0));
		}
	}
	const int num_output = this->layer_param_.convolution_param().num_output();
	C = num_output;
	const int axis = bottom[0]->CanonicalAxisIndex(
		this->layer_param_.convolution_param().axis());
	// Dimensions starting from "axis" are "flattened" into a single
	// length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
	// and axis == 1, N inner products with dimension CHW are performed.
	D = bottom[0]->count(axis);

	// parameter initialization
	center_ratio = this->layer_param_.convolution_param().bias_filler().std();
	lambda = 1 + center_ratio;
	strict_mode = this->layer_param_.convolution_param().bias_term();
	large_margin = this->layer_param_.dropout_param().dropout_ratio();
	// set centers
	float lr = this->layer_param_.convolution_param().bias_filler().value();
	float radius = this->layer_param_.convolution_param().bias_filler().mean()/sqrt(D) /** log(C+1) */;
	int iters = this->layer_param_.convolution_param().bias_filler().sparse();
	expand_center = this->layer_param_.convolution_param().bias_filler().min() > 0.01;
	update_center = this->layer_param_.convolution_param().bias_filler().max() > 0.01;
	LOG(INFO) << "strict_mode  : " << strict_mode;
	LOG(INFO) << "Center ratio : " << center_ratio;
	LOG(INFO) << "Lambda       : " << lambda;
	LOG(INFO) << "Margin       : " << large_margin;
	LOG(INFO) << "expand_center: " << expand_center;
	LOG(INFO) << "update_center: " << update_center;
	LOG(INFO) << "radius       : " << radius;
	LOG(INFO) << "iters        : " << iters;

	// Check if we need to set up the weights
	if (this->blobs_.size() > 0) {
		LOG(INFO) << "Skipping parameter initialization";
	}
	else 
	{
		this->blobs_.resize(1);
		// Intialize the weight
		vector<int> center_shape(2);
		center_shape[0] = C;
		center_shape[1] = D;
		this->blobs_[0].reset(new Blob<Dtype>(center_shape));
		// fill the weights
		shared_ptr<Filler<Dtype> > center_filler(GetFiller<Dtype>(
			this->layer_param_.convolution_param().weight_filler()));
		center_filler->Fill(this->blobs_[0].get());
		// expand center
		if (expand_center)
		{
			ExpandCenter<Dtype>(C, D, this->blobs_[0].get()->mutable_cpu_data(), lr, C * iters, radius);
		}
	}  
	
	this->param_propagate_down_.resize(this->blobs_.size(), update_center);
}

template <typename Dtype>
void MagneticLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom[1]->channels(), 1);
	CHECK_EQ(bottom[1]->height(), 1);
	CHECK_EQ(bottom[1]->width(), 1);
	M = bottom[0]->num(); // batch size
	// The top shape will be the bottom shape with the flattened axes dropped,
	// and replaced by a single axis with dimension num_output (N_).
	LossLayer<Dtype>::Reshape(bottom, top);
	// prob
	vector<int> prob_shape(2);
	prob_shape[0] = M;
	prob_shape[1] = C;
	prob.Reshape(prob_shape);
	// diff
	vector<int> diff_shape(3);
	diff_shape[0] = M;
	diff_shape[1] = C;
	diff_shape[2] = D;
	diff.Reshape(diff_shape);
	// dist
	dist.Reshape(M,C,1,1);
	// maxD, sumD
	maxD_.Reshape(M, 1, 1, 1);
	sumD_.Reshape(M, 1, 1, 1);
	// multiplier
	multiplier.Reshape(M, C, 1, 1);
	if (top.size() >= 2) {
		// softmax output
		top[1]->ReshapeLike(prob);
	}
	//LOG(INFO) << "-------------------- MCD:" << M << "," << C << "," << D;
}

template <typename Dtype>
void MagneticLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* label = bottom[1]->cpu_data();
	const Dtype* center = this->blobs_[0]->cpu_data();
	Dtype* diff_data = diff.mutable_cpu_data();
	Dtype* mutiplier_data = multiplier.mutable_cpu_data();
	Dtype* dist_data = dist.mutable_cpu_data();
	Dtype* prob_data = prob.mutable_cpu_data();

	Dtype loss = Dtype(0);
	//printf("\n");
	// the i-th distance_data
	for (int i = 0; i < M; i++) {
		const int Yi = static_cast<int>(label[i]);
		const Dtype* Xi = bottom_data + i * D;
		Dtype* pi = prob_data + i * C;
		Dtype* di = dist_data + i * C;
		Dtype sumD = Dtype(0);
		Dtype maxD = -1e10;
		Dtype* scale = mutiplier_data + i * C;
		// set multiplier
		caffe_set<Dtype>(C, -1, scale);
		if (this->phase_ == TRAIN)
		{
			scale[Yi] = -lambda;
		}
		// Xi - Ci
		for (int j = 0; j < C; j++)
		{
			const Dtype* Cj = center + j * D;
			Dtype* diff_j = diff_data + i*C*D + j*D;
			caffe_sub<Dtype>(D, Xi, Cj, diff_j);
			Dtype dj = caffe_cpu_dot<Dtype>(D, diff_j, diff_j);
			di[j] = 0.5 * dj;
		}
		// m * d -> d
		caffe_mul<Dtype>(C, scale, di, di);
		if (this->phase_ == TRAIN)
		{
			di[Yi] -= large_margin;
		}
		
		for (int j = 0; j < C; j++)
		{
			if (di[j] > maxD)
			{
				maxD = di[j];
			}
			//printf("%.3f ", di[j]);
		}
		// d -= maxD
		caffe_add_scalar<Dtype>(C, -maxD, di);
		// e^-dj
		caffe_exp(C, di, pi);
		// sum
		for (int j = 0; j < C; j++)
		{
			sumD += pi[j];
		}
		// prob
		for (int j = 0; j < C; j++)
		{
			pi[j] /= sumD;
			pi[j] = std::max(pi[j], Dtype(FLT_MIN));
			//printf("%.3f ", pi[j]);
		}
		//loss += -log(std::max(pi[Yi], Dtype(FLT_MIN)));
		loss += -log(pi[Yi]);
		//loss += (1-pi[Yi]);
		//printf("\n");
	}
	// average
	loss = loss / M;
	//loss = 1;
	//LOG(INFO) << "Loss:" << loss;
	top[0]->mutable_cpu_data()[0] = loss;
	if (top.size() == 2) {
		top[1]->ShareData(prob);
	}
}

template <typename Dtype>
void MagneticLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>*>& bottom) {
	const Dtype* label = bottom[1]->cpu_data();
	Dtype* mutiplier_data = multiplier.mutable_cpu_data();
	Dtype* prob_data = prob.mutable_cpu_data();
	caffe_copy(C*M, prob_data, mutiplier_data);
	for (int i = 0; i < M; i++) 
	{
		const int Yi = static_cast<int>(label[i]);
		Dtype* scale = mutiplier_data + i * C;
		scale[Yi] = lambda * (prob_data[i * C + Yi] - 1);
	}
	// Gradient with respect to centers
	if (update_center && this->param_propagate_down_[0])
	{
		const Dtype* label = bottom[1]->cpu_data();
		Dtype* center_diff = this->blobs_[0]->mutable_cpu_diff();
		const Dtype* diff_data = diff.cpu_data();
		Dtype* prob_data = prob.mutable_cpu_data();
		// Center update in Center Loss
		if (!strict_mode)
		{
			// \sum_{y_i==j}
			caffe_set(C * D, (Dtype)0., center_diff);
			for (int n = 0; n < C; n++) {
				int count = 0;
				for (int m = 0; m < M; m++) {
					const int Yi = static_cast<int>(label[m]);
					if (Yi == n) {
						count++;
						caffe_sub(D, center_diff + n * D, diff_data + m * D, center_diff + n * D);
					}
				}
				caffe_scal(D, (Dtype)1. / (count + (Dtype)1.), center_diff + n * D);
				//LOG(INFO) << center_diff[n * D];
			}
		}
		else
		{
			caffe_set(C * D, (Dtype)0., center_diff);
			for (int j = 0; j < C; j++) 
			{
				Dtype* Cj = center_diff + j * D;
				int count = 0;
				for (int i = 0; i < M; i++) 
				{
					const int Yi = static_cast<int>(label[i]);
					const Dtype* diff_j = diff_data + i*C*D + j*D;
					Dtype* pi = mutiplier_data + i * C;
					Dtype scalar = pi[j];
					if (Yi == j) 
					{
						count++;
					}

					caffe_axpy(D, scalar, diff_j, Cj);
				}
				//caffe_scal(D, (Dtype)2. / (count + (Dtype)1.), Cj);
				caffe_scal(D, (Dtype)1. / (Dtype)M, Cj);
			}
		}
	}
	// Gradient with respect to bottom data 
	if (propagate_down[0])
	{
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* label = bottom[1]->cpu_data();

		Dtype* diff_data = diff.mutable_cpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		// Zero
		caffe_set(M * D, (Dtype)0., bottom_diff);
		for (int i = 0; i < M; i++) 
		{
			const int Yi = static_cast<int>(label[i]);
			const Dtype* Xi = bottom_data + i * D;
			Dtype* dXi = bottom_diff + i * D;
			Dtype* pi = mutiplier_data + i * C;
			// sum_{j}pj(Xi - Cj)
			for (int j = 0; j < C; j++)
			{
				Dtype* diff_j = diff_data + i*C*D + j*D;
				caffe_axpy(D, -pi[j], diff_j, dXi);
			}
		}
		caffe_scal(M * D, top[0]->cpu_diff()[0] / M, bottom[0]->mutable_cpu_diff());
	}
	if (propagate_down[1]) {
		LOG(FATAL) << this->type()
			<< " Layer cannot backpropagate to label inputs.";
	}
}

#ifdef CPU_ONLY
STUB_GPU(MagneticLossLayer);
#endif

INSTANTIATE_CLASS(MagneticLossLayer);
REGISTER_LAYER_CLASS(MagneticLoss);

}  // namespace caffe
