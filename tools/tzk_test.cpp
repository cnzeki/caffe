#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;

// A simple registry for caffe commands.
typedef int(*BrewFunction)();
typedef std::map<caffe::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

#define RegisterBrewFunction(func) \
namespace { \
\
class __Registerer_##func{ \
public: /* NOLINT */ \
	__Registerer_##func() { \
	\
	g_brew_map[#func] = &func; \
	} \
}; \
__Registerer_##func g_registerer_##func; \
}

static BrewFunction GetBrewFunction(const caffe::string& name) {
	if (g_brew_map.count(name)) {
		return g_brew_map[name];
	}
	else {
		LOG(ERROR) << "Available caffe actions:";
		for (BrewMap::iterator it = g_brew_map.begin();
			it != g_brew_map.end(); ++it) {
			LOG(ERROR) << "\t" << it->first;
		}
		LOG(FATAL) << "Unknown action: " << name;
		return NULL;  // not reachable, just to suppress old compiler warnings.
	}
}


#include <float.h>
#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

template<typename DataType>
static void minmax(int n, DataType* data, DataType* dstMin, DataType* dstMax)
{
	DataType min, max;
	if (!data)
	{
		return;
	}

	min = data[0];
	max = data[0];
	for (int i = 1; i < n; i++)
	{
		min = MIN(min, data[i]);
		max = MAX(max, data[i]);
	}

	if (dstMin)
	{
		*dstMin = min;
	}
	if (dstMax)
	{
		*dstMax = max;
	}
}

static void cvDrawText(IplImage *image, CvPoint point, const char* str, CvScalar color = cvScalar(255, 255, 255), float size = 1.0)
{
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, size, size, 0, cvRound(size * 2), 8);
	cvPutText(image, str, point, &font, color);
}

template <typename Dtype>
void writeBlobFile(const char* path, const Blob<Dtype>* blob)
{
	FILE* file = fopen(path, "w");
	const Dtype* data = blob->cpu_data();
	int planeCount = blob->count() / blob->num();
	for (int i = 0; i < blob->num(); i++)
	{
		const Dtype* plane = data + blob->offset(i);
		for (int j = 0; j < planeCount; j++)
		{
			fprintf(file, "%.2f,", plane[j]);
		}
		fprintf(file, "\n");
	}
	fclose(file);
}

// Visualize a blob to a image. Max abs weight mapped to 255.Green for positive weights
// Red for negative ones.Filters in row order,channels in colnum order.
template <typename Dtype>
IplImage* createBlobImage(const Blob<Dtype>* blob, int binSize = 1)
{
	const Dtype* data = blob->cpu_data();
	Dtype maxV = FLT_MIN;
	for (int i = 0; i < blob->count(); i++)
	{
		maxV = MAX(maxV, abs(data[i]));
	}
	float scalar = 255.0f / (float)maxV;
	IplImage* image = cvCreateImage(cvSize(blob->channels()*blob->width()*binSize,
		blob->num() * blob->height() * binSize),
		IPL_DEPTH_8U,
		3);

	for (int n = 0; n < blob->num(); n++)
	{
		for (int c = 0; c < blob->channels(); c++)
		{
			for (int h = 0; h < blob->height(); h++)
			{
				for (int w = 0; w < blob->width(); w++)
				{
					Dtype amp = data[blob->offset(n, c, h, w)];
					CvScalar color;
					if (amp > 0)
					{
						color = cvScalar(0, amp * scalar, 0);
					}
					else
					{
						color = cvScalar(0, 0, abs(amp)*scalar);
					}

					cvRectangleR(image,
						cvRect(c*blob->width()*binSize + w * binSize,
						n*blob->height()*binSize + h * binSize,
						binSize, binSize),
						color,
						-1);
				}
			}
			cvLine(image, cvPoint(c*blob->width()*binSize, 0), cvPoint(c*blob->width()*binSize, image->height), cvScalar(200, 200, 200), 1);
		}

		cvLine(image, cvPoint(0, n*blob->height()*binSize), cvPoint(image->width, n*blob->height()*binSize), cvScalar(200, 200, 200), 1);
	}

	return image;
}

template <typename Dtype>
void writeBlobImage(const char* path, const Blob<Dtype>* blob, int binSize = 2)
{
	IplImage* image = createBlobImage(blob, binSize);
	cvSaveImage(path, image);
	cvReleaseImage(&image);
}

template <typename Dtype>
void displayNetStructure(Net<Dtype>& net)
{
	const vector<boost::shared_ptr<Layer<float> > >& layers = net.layers();
	vector<vector<Blob<float>*> >& bottomVecs = net.bottom_vecs();
	vector<vector<Blob<float>*> >& topVecs = net.top_vecs();

	printf("Net : %s\n", net.name().c_str());
	// layers
	printf("Layers:%d\n", layers.size());
	printf("------------------------------------------\n");
	for (int i = 0; i < layers.size(); i++)
	{
		const LayerParameter& param = layers[i]->layer_param();
		vector<Blob<float>*> bottomVec = bottomVecs[i];
		vector<Blob<float>*> topVec = topVecs[i];

		printf("Layer[%d] name:%s type:%s \n", i, param.name().c_str(), LayerParameter::LayerType_Name(param.type()).c_str());

		printf("\tbottom_blobs:%d\n", bottomVec.size());
		for (int j = 0; j < bottomVec.size(); j++)
		{
			printf("\t[%d] name:%s dims[%d,%d,%d,%d]\n", j, param.bottom(j).c_str(),
				bottomVec[j]->num(),
				bottomVec[j]->channels(),
				bottomVec[j]->width(),
				bottomVec[j]->height());
		}
		printf("\ttop_blobs:%d\n", topVec.size());
		for (int j = 0; j < topVec.size(); j++)
		{
			printf("\t[%d] name:%s dims[%d,%d,%d,%d]\n", j, param.top(j).c_str(),
				topVec[j]->num(),
				topVec[j]->channels(),
				topVec[j]->width(),
				topVec[j]->height());
		}

		vector<boost::shared_ptr<Blob<Dtype> > >& layer_blobs = layers[i]->blobs();
		printf("\tparam blobs:%d\n", layer_blobs.size());
		for (int j = 0; j < layer_blobs.size(); ++j) {
			printf("\t[%d] dims[%d,%d,%d,%d]\n", j,
				layer_blobs[j]->num(),
				layer_blobs[j]->channels(),
				layer_blobs[j]->width(),
				layer_blobs[j]->height());
			if (param.type() == LayerParameter_LayerType_CONVOLUTION && 0 == j)
			{
				char path[256];
				sprintf(path, "e:/layer_%d-%d.csv", i, j);
				writeBlobFile<float>(path, (layer_blobs[j]).get());
				sprintf(path, "e:/layer_%d-%d.bmp", i, j);
				writeBlobImage<float>(path, (layer_blobs[j]).get(), 4);
			}
		}
		printf("------------------------------------------\n");
	}
}

void debug_mnist_cnn(Net<float>& net, int& error_count, const char* save_result_prefix = NULL)
{
	// get blob by name
	const boost::shared_ptr<Blob<float>> blob_data = net.blob_by_name("data");
	const boost::shared_ptr<Blob<float>> blob_label = net.blob_by_name("label");
	const boost::shared_ptr<Blob<float>> blob_prob = net.blob_by_name("prob");

	// num of samples
	int num = blob_data->num();
	int pixel_size = 2;
	IplImage* img_data = createBlobImage<float>(blob_data.get(), pixel_size);
	int batch_height = img_data->height / num;
	int prob_num = blob_prob->channels();
	int prob_xsize = 5;
	int right_margin = 200;
	IplImage* img_prob = cvCreateImage(cvSize(prob_num*prob_xsize + right_margin, batch_height*num), IPL_DEPTH_8U, 3);
	cvZero(img_prob);

	int errors = 0;
	const float* ptr_label = blob_label->cpu_data();
	const float* ptr_prob = blob_prob->cpu_data();
	for (int i = 0; i < num; i++)
	{
		int label = (int)ptr_label[blob_label->offset(i)];
		int max_idx = 0;
		float max_prob = 0;
		for (int j = 0; j < prob_num; j++)
		{
			float prob = ptr_prob[blob_prob->offset(i, j)];
			if (prob > max_prob)
			{
				max_prob = prob;
				max_idx = j;
			}
		}

		for (int j = 0; j < prob_num; j++)
		{
			float prob = ptr_prob[blob_prob->offset(i, j)];
			CvScalar color = cvScalar(0, 0, 255);
			if (j == max_idx)
			{
				color = cvScalar(0, 255, 255);
			}
			if (j == label)
			{
				color = cvScalar(0, 255, 0);
			}

			cvRectangleR(img_prob,
				cvRect(j*prob_xsize, i*batch_height, prob_xsize, prob*batch_height),
				color,
				-1);

			cvLine(img_prob, cvPoint(j*prob_xsize, i*batch_height),
				cvPoint(j*prob_xsize, (i + 1)*batch_height), cvScalar(200, 200, 200), 1);

		}
		char text[200];
		sprintf(text, "%d - %d", label, max_idx);
		CvScalar color = cvScalar(0, 255, 255);
		if (label == max_idx)
		{
			color = cvScalar(0, 255, 0);
		}
		else
		{
			errors++;
		}
		cvDrawText(img_prob, cvPoint(prob_num*prob_xsize + 20, i*batch_height + 30), text, color);
		cvLine(img_prob, cvPoint(0, i*batch_height),
			cvPoint(prob_num*prob_xsize, i*batch_height), cvScalar(200, 200, 200), 1);
	}

	// display in one image
	IplImage* canvas = cvCreateImage(cvSize(img_prob->width + img_data->width, img_data->height), IPL_DEPTH_8U, 3);

	cvZero(canvas);
	cvSetImageROI(canvas, cvRect(0, 0, img_data->width, img_data->height));
	cvCopy(img_data, canvas);
	cvResetImageROI(canvas);

	cvSetImageROI(canvas, cvRect(img_data->width, 0, img_prob->width, img_prob->height));
	cvCopy(img_prob, canvas);
	cvResetImageROI(canvas);

	if (save_result_prefix)
	{
		char path[300];
		sprintf(path, "%s%d.png", save_result_prefix, error_count + 1/*,error_count+errors*/);
		cvSaveImage(path, canvas);
	}
	else
	{
		cvShowImage("data_prob", canvas);
		cvWaitKey();
	}

	cvReleaseImage(&canvas);
	cvReleaseImage(&img_prob);
	cvReleaseImage(&img_data);
	// add errors
	error_count += errors;
}

void debug_mnist_autoencoder(Net<float>& net, int& error_count, const char* save_result_prefix = NULL)
{
	// get blob by name
	const boost::shared_ptr<Blob<float>> blob_data = net.blob_by_name("data");
	const boost::shared_ptr<Blob<float>> blob_prob = net.blob_by_name("decode1neuron");
	int num = blob_data->num();
	Blob<float> reconstuct;
	reconstuct.ReshapeLike(*blob_data);
	reconstuct.ShareData(*blob_prob);
	int binSize = 2;
	IplImage* img_data = createBlobImage<float>(blob_data.get(), binSize);
	IplImage* img_prob = createBlobImage<float>(&reconstuct, binSize);
	// display in one image
	IplImage* canvas = cvCreateImage(cvSize(img_prob->width + img_data->width, img_data->height), IPL_DEPTH_8U, 3);

	cvZero(canvas);
	cvSetImageROI(canvas, cvRect(0, 0, img_data->width, img_data->height));
	cvCopy(img_data, canvas);
	cvResetImageROI(canvas);

	cvSetImageROI(canvas, cvRect(img_data->width, 0, img_prob->width, img_prob->height));
	cvCopy(img_prob, canvas);
	cvResetImageROI(canvas);
	cvShowImage("data_prob", canvas);
	cvWaitKey();

	cvReleaseImage(&canvas);
	cvReleaseImage(&img_prob);
	cvReleaseImage(&img_data);
}

bool is_image_region_conflict(IplImage* canvas, CvRect rect, IplImage* img,
	CvScalar bkColor, float min_visible_value, float conflict_ratio)
{
	// check this region can contain img
	int conflict_pixels = 0;
	int img_pixels = 0;
	uchar color[3];
	color[0] = bkColor.val[0];
	color[1] = bkColor.val[1];
	color[2] = bkColor.val[2];
	for (int r = 0; r < img->height; r++)
	{
		float* img_ptr = (float*)(img->imageData + img->widthStep*r);
		uchar* canvas_ptr = (uchar*)(canvas->imageData + (rect.y + r)*canvas->widthStep + rect.x * 3);
		for (int c = 0; c < img->width; c++)
		{
			if (img_ptr[0] > min_visible_value )
			{
				if (canvas_ptr[0] != color[0] || canvas_ptr[1] != color[1] || canvas_ptr[2] != color[2])
				{
					conflict_pixels++;
				}
				img_pixels++;
			}

			img_ptr += 1;
			canvas_ptr += 3;
		}
	}
	// conflict ratio
	return conflict_pixels > img_pixels * conflict_ratio;
}

void image_overlay(IplImage* canvas, CvRect rect, IplImage* img,
	float min_visible_value, CvScalar color)
{
	for (int r = 0; r < img->height; r++)
	{
		float* img_ptr = (float*)(img->imageData + img->widthStep*r);
		uchar* canvas_ptr = (uchar*)(canvas->imageData + (rect.y + r)*canvas->widthStep + rect.x * 3);
		for (int c = 0; c < img->width; c++)
		{
			if (img_ptr[0] > min_visible_value)
			{
				int a = 255 * (1.0f - img_ptr[0]);
				canvas_ptr[0] = color.val[0] * img_ptr[0] + a;
				canvas_ptr[1] = color.val[1] * img_ptr[0] + a;
				canvas_ptr[2] = color.val[2] * img_ptr[0] + a;
			}

			img_ptr += 1;
			canvas_ptr += 3;
		}
	}
}

static int  RandInt(int min, int max)
{
	int v = rand();
	int ret = v % (max - min) + min;
	return ret;

}

IplImage* render_image_xy(vector<float>& X,
	vector<float>& Y,
	vector<IplImage*>&      images,
	vector<int>&            labels)
{
	float xmax, xmin, ymax, ymin;
	minmax<float>(X.size(), &X[0], &xmin, &xmax);
	minmax<float>(Y.size(), &Y[0], &ymin, &ymax);
	float xrange, yrange;
	xrange = xmax - xmin;
	yrange = ymax - ymin;
	float max_range = MAX(xrange, yrange);
	float scalar = 1000.0f / max_range;
	int border = 100;
	int xsize = scalar * xrange + border;
	int ysize = scalar * yrange + border;
	IplImage* canvas = cvCreateImage(cvSize(xsize, ysize), IPL_DEPTH_8U, 3);
	// white background
	cvSet(canvas, cvScalar(255, 255, 255));
	// colors for digits
	CvScalar colors[10];
	// random color
	for (int i = 0; i < 10; i++)
	{
		colors[i] = cvScalar(RandInt(0, 255), RandInt(0, 200), RandInt(50, 255));
	}
	// random shuffle samples
	vector<int> indexs(X.size());
	for (int i = 0; i < X.size(); i++)
	{
		indexs[i] = i;
	}
	random_shuffle(indexs.begin(), indexs.end());
	int visualize_count = X.size();
	for (int i = 0; i < visualize_count; i++)
	{
		int id = indexs[i];
		int label = labels[id];
		float x = (X[id] - xmin)*scalar + border / 2;
		float y = (Y[id] - ymin)*scalar + border / 2;
		IplImage* img = images[id];

		CvRect rect = cvRect(x - img->width / 2, y - img->height / 2, img->width, img->height);
		// check this region can contain img
		int conflict_pixels = 0;
		float min_visible_value = 0.1;
		float conflict_ratio = 0.2;
		// if not conflict, render img to canvas
		if (!is_image_region_conflict(canvas, rect, img, cvScalar(255, 255, 255), min_visible_value, conflict_ratio))
		{
			image_overlay(canvas, rect, img, min_visible_value, colors[label]);
		}
	}

	return canvas;
}

IplImage* render_code2_image(vector<vector<double>>& features,
	vector<IplImage*>&      images,
	vector<int>&            labels)
{
	vector<float> X(features.size());
	vector<float> Y(features.size());
	for (int i = 0; i < features.size(); i++)
	{
		X[i] = features[i][0];
		Y[i] = features[i][1];
	}

	return render_image_xy(X, Y, images, labels);
}

IplImage* calc_image_on_pca(vector<IplImage*>&      images,
	vector<int>&            labels)
{
	int numSamples = images.size();
	int dataDim = images[0]->width*images[0]->height;
	// PCA
	int maxComponets = 2;
	Mat trainMat(numSamples, dataDim, CV_32FC1);
	for (int i = 0; i < numSamples; i++)
	{
		memcpy(trainMat.ptr<float>(i), images[i]->imageData, sizeof(float)*dataDim);
	}
	PCA pca(trainMat, Mat(), CV_PCA_DATA_AS_ROW, maxComponets);
	Mat compressed(numSamples, maxComponets, CV_32FC1);
	pca.project(trainMat, compressed);

	// Get PCA coefficients feature
	vector<float> X(numSamples);
	vector<float> Y(numSamples);
	for (int i = 0; i < numSamples; i++)
	{
		X[i] = compressed.at<float >(i, 0);
		Y[i] = compressed.at<float>(i, 1);
	}

	return render_image_xy(X, Y, images, labels);
}

template <typename Dtype>
void writeVectorArray(const char* path, vector<vector<Dtype>>& features, const char* fmt_str)
{
	FILE* file = fopen(path, "w");
	for (int i = 0; i < features.size(); i++)
	{
		vector<double>& f = features[i];
		for (int j = 0; j < f.size(); j++)
		{
			fprintf(file, fmt_str, f[j]);
		}
		fprintf(file, "\n");
	}
	fclose(file);
}

void debug_mnist_autoencoder_som(Net<float>& net, int total_iter)
{
	vector<vector<double>> features;
	vector<IplImage*>      images;
	vector<int>            labels;
	// collect all features
	for (int i = 0; i < total_iter; ++i) {
		net.ForwardPrefilled();
		// get blob by name
		const boost::shared_ptr<Blob<float>> blob_data = net.blob_by_name("data");
		const boost::shared_ptr<Blob<float>> blob_label = net.blob_by_name("label");
		const boost::shared_ptr<Blob<float>> blob_feat = net.blob_by_name("encode4");

		int num = blob_data->num();
		int feat_dim = blob_feat->count() / num;
		for (int s = 0; s < num; s++)
		{
			// feature
			vector<double> feat(feat_dim);
			for (int f = 0; f < feat_dim; f++)
			{
				feat[f] = blob_feat->data_at(s, f, 0, 0);
			}
			// data
			IplImage* img = cvCreateImage(cvSize(blob_data->width(), blob_data->height()), IPL_DEPTH_32F, 1);
			float* img_ptr = (float*)img->imageData;
			const float* sample_ptr = blob_data->cpu_data() + blob_data->offset(s);
			memcpy(img_ptr, sample_ptr, sizeof(float)*blob_data->offset(1));

			// label
			int label = (int)blob_label->data_at(s, 0, 0, 0);
			features.push_back(feat);
			images.push_back(img);
			labels.push_back(label);
		}

		LOG(ERROR) << "Batch " << i;
	}

	IplImage* canvas;
	canvas = render_code2_image(features, images, labels);
	//render_som_image(features, images, labels);	
	//canvas = calc_som_on_pixel(images, labels);
	//canvas = calc_som_on_pca(images, labels);
	//canvas = calc_image_on_pca(images, labels);
	//canvas = render_som_image(features, images, labels);
	cvSaveImage("e:/mnist_pca_30components.png", canvas);
	cvReleaseImage(&canvas);

	// write features to csv
	// writeVectorArray<double>("e:/mnist_code30.csv",features, "%.2lf,");

	// release images
	for (int i = 0; i < images.size(); i++)
	{
		cvReleaseImage(&images[i]);
	}
}

void debug_mnist_siamese(Net<float>& net, int total_iter, const char* img_path)
{
	vector<vector<double>> features;
	vector<IplImage*>      images;
	vector<int>            labels;
	// collect all features
	for (int i = 0; i < total_iter; ++i) {
		net.ForwardPrefilled();
		// get blob by name
		const boost::shared_ptr<Blob<float>> blob_data = net.blob_by_name("data");
		const boost::shared_ptr<Blob<float>> blob_label = net.blob_by_name("label");
		const boost::shared_ptr<Blob<float>> blob_feat = net.blob_by_name("ip2");

		int num = blob_data->num();
		int feat_dim = blob_feat->count() / num;
		for (int s = 0; s < num; s++)
		{
			// feature
			vector<double> feat(feat_dim);
			for (int f = 0; f < feat_dim; f++)
			{
				feat[f] = blob_feat->data_at(s, f, 0, 0);
			}
			// data
			IplImage* img = cvCreateImage(cvSize(blob_data->width(), blob_data->height()), IPL_DEPTH_32F, 1);
			float* img_ptr = (float*)img->imageData;
			const float* sample_ptr = blob_data->cpu_data() + blob_data->offset(s);
			memcpy(img_ptr, sample_ptr, sizeof(float)*blob_data->offset(1));

			// label
			int label = (int)blob_label->data_at(s, 0, 0, 0);
			features.push_back(feat);
			images.push_back(img);
			labels.push_back(label);
		}

		LOG(ERROR) << "Batch " << i;
	}

	IplImage* canvas;
	canvas = render_code2_image(features, images, labels);
	//render_som_image(features, images, labels);	
	//canvas = calc_som_on_pixel(images, labels);
	//canvas = calc_som_on_pca(images, labels);
	//canvas = calc_image_on_pca(images, labels);
	//canvas = render_som_image(features, images, labels);
	cvSaveImage(img_path, canvas);
	cvReleaseImage(&canvas);

	// write features to csv
	//writeVectorArray<double>("e:/mnist_siamese_2.csv",features, "%.2lf,");

	// release images
	for (int i = 0; i < images.size(); i++)
	{
		cvReleaseImage(&images[i]);
	}
}

DEFINE_int32(gpu, -1,
	"Run in GPU mode on given device ID.");
DEFINE_string(dir, "",
	"The solver definition protocol buffer text file.");
DEFINE_int32(iters, 100,
	"The number of iterations to run.");
DEFINE_int32(step, 1000,
	"The number of iterations to run.");
DEFINE_int32(count, 10,
	"The number of iterations to run.");
DEFINE_string(model, "",
	"The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
	"Optional; the snapshot solver state to resume training.");
DEFINE_string(weights, "",
	"Optional; the pretrained weights to initialize finetuning. "
	"Cannot be set simultaneously with snapshot.");



void test_model(std::string model, std::string weight, int iters, const char* img_path)
{
	Net<float> caffe_net(model, caffe::TEST);
	caffe_net.CopyTrainedLayersFrom(weight);
	LOG(INFO) << "Running for " << FLAGS_iters << " iterations.";

	debug_mnist_siamese(caffe_net, iters, img_path);
}

#include <stdarg.h>  
static inline std::string formatstr(const char *fmt, ...)
{
#define FORMAT_MSG_BUFFER_SIZE (2048)  
	char szBuffer[FORMAT_MSG_BUFFER_SIZE + 1] = { 0 };
	va_list args;
	va_start(args, fmt);
	vsnprintf(szBuffer, FORMAT_MSG_BUFFER_SIZE, fmt, args);
	va_end(args);
	std::string strRet = szBuffer;
	return strRet;
}

void test_batch()
{
	std::string model_dir = FLAGS_dir;
	int iterations = FLAGS_iters;
	for (int i = 1; i <= FLAGS_count; i++)
	{
		std::string mname = formatstr("exp_iter_%d",i*FLAGS_step);
		std::string weight = model_dir + mname+".caffemodel";
		std::string net_proto = model_dir + "lenet_train_test.prototxt";
		std::string img_path = mname + ".png";
		test_model(net_proto,weight,iterations,img_path.c_str());
	}
}

int main(int argc, char** argv) {
	// Print output to stderr (while still logging).
	FLAGS_alsologtostderr = 1;
	
	caffe::GlobalInit(&argc, &argv);

	// Set device id and mode
	if (FLAGS_gpu >= 0) {
		LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
		Caffe::SetDevice(FLAGS_gpu);
		Caffe::set_mode(Caffe::GPU);
	}
	else {
		LOG(INFO) << "Use CPU.";
		Caffe::set_mode(Caffe::CPU);
	}

	test_batch();
	return 0;
}