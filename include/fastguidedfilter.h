// Source: https://github.com/Sundrops/fast-guided-filter
// Literature: https://arxiv.org/pdf/1505.00996.pdf

#ifndef GUIDED_FILTER_H
#define GUIDED_FILTER_H

#include <opencv2/opencv.hpp>

class FastGuidedFilterImpl;

class FastGuidedFilter
{
public:
    FastGuidedFilter(const cv::Mat &I, int r, double eps,int s);
    ~FastGuidedFilter();

    cv::Mat __attribute__((target(mic))) filter(const cv::Mat &p, int depth = -1) const;

private:
    __attribute__((target(mic))) FastGuidedFilterImpl *impl_;
};

cv::Mat fastGuidedFilter(const cv::Mat &I, const cv::Mat &p, int r, double eps, int s = 1,int depth = -1);

#endif
