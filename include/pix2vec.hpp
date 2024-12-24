//Copyright (c) [2024] [Initial_Equationor]
//[pix2vec] is licensed under Mulan PubL v2.
//You can use this software according to the terms and conditions of the Mulan PubL v2.
//You may obtain a copy of Mulan PubL v2 at:
//http://license.coscl.org.cn/MulanPubL-2.0
//THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
//EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
//MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//See the Mulan PubL v2 for more details.
#pragma once
#include <vector>

namespace Pix2Vec {
typedef struct {
    int x;
    int y;
} Point;

typedef struct Ring{
    int nPoints;
    Point* points;
} Ring;

typedef struct {
    int nInnerRings;
    Ring* innerRing;
    Ring outerRing;
} Polygon;

typedef struct {
    int nPolygons;
    Polygon* polygons;
} MultiPolygon;

/// padding一个像素，zero-padding
/// \param width
/// \param height 值化图像 1-主体 0-背景
/// \param input
/// \param output
void Padding(int width, int height, const bool *input, bool* output);

/// 检测边缘像素
/// \param width
/// \param height
/// \param input padding后的二值化图像 1-主体 0-背景
/// \param output 边缘检测结果 1-边缘像素 0-不是边缘像素
void ExtractEdges(int width, int height, const bool *input, bool* output);

/// 追踪边界
/// \param width
/// \param height
/// \param origin padding后的原始二值化图像 1-主体 0-背景
/// \param edge 边缘检测结果 1-边缘像素 0-不是边缘像素
/// \return 所有边界，不分内环外环
std::vector<Ring> ExtractRings(int width, int height, const bool *origin, bool* edge);

/// 单个边缘抽稀，道格拉斯-普克算法实现，抽稀后点数量<3的环应当直接舍弃
/// \param origin 未抽稀的环
/// \param origin 抽稀后的结果，不需要提前初始化，记得手动释放output.points
/// \param maxDistThres 道格拉斯-普克算法距离阈值[像素]
void RingSimplifyDP(const Ring &origin, Ring &output, float maxDistThres= 1.5f);

/// 检查环的走向是顺时针还是逆时针
/// \param ring
/// \return 1-顺时针 0-逆时针
bool IsClockwise(const Ring& ring);

/// 点是否包含在多边形内，射线法检测
/// \param p
/// \param ring
/// \return 1-包含 0-不包含
bool IsPointInRing(const Point& p, const Ring& ring);

/// 构建MultiPolygon结构
/// \param outerRings 抽稀后的外环（没抽稀后的也可以，但计会增加计算复杂度）
/// \param innerRings 抽稀后的内环（没抽稀后的也可以，但计会增加计算复杂度）
/// \param output 不需要初始化，MultiPolygon.polygons、Polygon.innerRing和Ring.points
void BuildMultiPolygon(const std::vector<Ring> &outerRings, const std::vector<Ring> &innerRings, MultiPolygon& output);

/// 从原始二值化图像构建MultiPolygon
/// \param width
/// \param height
/// \param input
/// \param output
/// \param maxDistThres
void Vecterize(int width, int height, bool* input, MultiPolygon& output, float maxDistThres=1.5f);

void FreePolygon(Polygon& polygon);
void FreeMultiPolygon(MultiPolygon& multiPolygon);
} // namespace Pix2Vec
