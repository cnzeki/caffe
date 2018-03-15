#ifndef CAFFE_MAGNETIC_LOSS_LAYER_HPP_
#define CAFFE_MAGNETIC_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class MagneticLossLayer : public LossLayer<Dtype> {
 public:
	 explicit MagneticLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MagneticLoss"; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M;   // #batch size
  int C;   // #classes
  int D;   // #feature dims
  
  Blob<Dtype> prob;		// MxC
  Blob<Dtype> diff;		// MxCxD
  Blob<Dtype> dist;		// MxC
  Blob<Dtype> multiplier;// MxC
  Blob<Dtype> maxD_;	// Mx1
  Blob<Dtype> sumD_;	// Mx1
  // param
  Dtype  center_ratio;
  Dtype  lambda;
  Dtype  large_margin;
  bool   strict_mode;
  bool   expand_center;
  bool   update_center;
};

}  // namespace caffe

#endif  // CAFFE_MAGNETIC_LOSS_LAYER_HPP_