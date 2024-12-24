//Copyright (c) [2024] [Initial_Equationor]
//[pix2vec] is licensed under Mulan PubL v2.
//You can use this software according to the terms and conditions of the Mulan PubL v2.
//You may obtain a copy of Mulan PubL v2 at:
//http://license.coscl.org.cn/MulanPubL-2.0
//THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
//EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
//MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//See the Mulan PubL v2 for more details.
#include "pix2vec.hpp"
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <CL/opencl.hpp>

namespace Pix2Vec {
static cl::Platform g_clPlatform;
static std::vector<cl::Device> g_clDevices;
static cl::Context g_clContext;
static cl::CommandQueue g_clQueue;
static cl::Program g_clProgram;
static bool g_clEnvInitialized = false;

void initClEnv() {
    if (g_clEnvInitialized) return;
    // 初始化 OpenCL 资源
    cl::Platform::get(&g_clPlatform);
    g_clPlatform.getDevices(CL_DEVICE_TYPE_GPU, &g_clDevices);
    if (g_clDevices.empty()) {
        std::cerr << "[terrain_builder.cpp]clCollisionDetection: fail to load gpu device, using cpu." << std::endl;
        g_clPlatform.getDevices(CL_DEVICE_TYPE_CPU, &g_clDevices);
    }
    g_clContext = cl::Context(g_clDevices[0]);
    g_clQueue = cl::CommandQueue(g_clContext, g_clDevices[0]);
    if (!std::filesystem::exists("kernel/pix2vec.cl")) {
        throw std::runtime_error("[Pix2Vec]initClEnv: kernel source not exist!");
    }
    std::ifstream file("kernel/pix2vec.cl");
    std::stringstream buffer;
    buffer << file.rdbuf();
//    std::cout << buffer.str() << std::endl;
    cl_int err;
    g_clProgram = cl::Program(g_clDevices, buffer.str(), true, &err);
    if (err != CL_SUCCESS) {
        std::stringstream _ss;
        _ss << "[Pix2Vec]initClEnv: load source error, ";
        std::string build_log;
        err = g_clProgram.getBuildInfo(g_clDevices[0], CL_PROGRAM_BUILD_LOG, &build_log);
        if (err == CL_SUCCESS) {
            _ss << "Build log for device: " << build_log;
        } else {
            _ss << "Error getting build log.";
        }
        throw std::runtime_error(_ss.str());
    }
    err = g_clProgram.build(g_clDevices[0]);
    if (err != CL_SUCCESS) {
        std::string buildLog;
        g_clProgram.getBuildInfo(g_clDevices[0], CL_PROGRAM_BUILD_LOG, &buildLog);
        std::stringstream _ss;
        _ss << "[Pix2Vec]initClEnv: Build log for device: " << g_clDevices[0].getInfo<CL_DEVICE_NAME>() << std::endl;
        _ss << buildLog << std::endl;
        throw std::runtime_error(_ss.str());
    }
    g_clEnvInitialized = true;
} // initClEnv

void Padding(int width, int height, const bool *input, bool* output) {
    memset(output, 0, (width+2)*(height+2)*sizeof(bool));
    for (int y = 0; y < height; ++y) {
        memcpy(&output[(width+2)*(y+1)+1], &input[width*y], width * sizeof(bool));
    }
} // void Padding

void ExtractEdges(int width, int height, const bool *input, bool* output) {
    initClEnv();
    cl::Buffer inputBuffer(g_clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           (width+2)*(height+2)*sizeof(cl_uchar), (cl_uchar*)input);
    cl::Buffer outputBuffer(g_clContext, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                            width*height*sizeof(cl_uchar), output);
    cl::Kernel kernel(g_clProgram, "extract_edges");
    kernel.setArg(0, width);
    kernel.setArg(1, height);
    kernel.setArg(2, inputBuffer);
    kernel.setArg(3, outputBuffer);
    cl::NDRange globalWorkSize(width, height);
    cl_uint err = g_clQueue.enqueueNDRangeKernel(kernel, cl::NullRange, globalWorkSize, cl::NullRange);
    g_clQueue.finish();
    g_clQueue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, width*height*sizeof(cl_uchar), output);
} // void ExtractEdges

Ring traceEdge(int width, int height, int beginX, int beginY, const bool *origin, bool* openset) {
    std::vector<Point> points;
    openset[width * beginY + beginX] = false;
    // 确定起始点
    if ( !(origin[(width+2)*beginY+beginX]
        && origin[(width+2)*beginY+beginX+1]
        && origin[(width+2)*(beginY+1)+beginX])) {
        // 左上
        points.push_back(Point{ beginX, beginY });
    } else if (!(origin[(width+2)*beginY+beginX+1]
              && origin[(width+2)*beginY+beginX+2]
              && origin[(width+2)*(beginY+1)+beginX+2])) {
        // 右上
        points.push_back(Point{ beginX+1, beginY });
    } else if (!(origin[(width+2)*(beginY+1)+beginX+2]
              && origin[(width+2)*(beginY+2)+beginX+1]
              && origin[(width+2)*(beginY+2)+beginX+2])) {
        // 右下
        points.push_back(Point{ beginX+1, beginY+1 });
    } else if (!(origin[(width+2)*(beginY+1)+beginX]
              && origin[(width+2)*(beginY+2)+beginX]
              && origin[(width+2)*(beginY+2)+beginX+1])) {
        // 左下
        points.push_back(Point{ beginX, beginY+1 });
    }
    // first step
    Point _cur = points.back();
    Point _next = _cur;
    if (!origin[(width+2)*_cur.y+_cur.x]     && origin[(width+2)*_cur.y+_cur.x+1]
      && origin[(width+2)*(_cur.y+1)+_cur.x] && !origin[(width+2)*(_cur.y+1)+_cur.x+1]) {
        if (beginX == _cur.x) {
            _next.y --;
        } else {
            _next.y ++;
        }
    } else if (origin[(width+2)*_cur.y+_cur.x]     && !origin[(width+2)*_cur.y+_cur.x+1]
           && !origin[(width+2)*(_cur.y+1)+_cur.x] && origin[(width+2)*(_cur.y+1)+_cur.x+1]) {
        if (beginY == _cur.y) {
            _next.x ++;
        } else {
            _next.x --;
        }
    } else if (!origin[(width+2)*_cur.y+_cur.x] && origin[(width+2)*_cur.y+_cur.x+1]) {
        _next.y --;
        openset[width * (_cur.y - 1) + _cur.x] = false;
        if (origin[(width+2)*(_cur.y+1)+_cur.x+1]) {
            openset[width * _cur.y + _cur.x] = false;
        }
    } else if (!origin[(width+2)*_cur.y+_cur.x+1] && origin[(width+2)*(_cur.y+1)+_cur.x+1]) {
        _next.x ++;
        openset[width * _cur.y + _cur.x] = false;
        if (origin[(width+2)*(_cur.y+1)+_cur.x]) {
            openset[width * _cur.y + _cur.x - 1] = false;
        }
    } else if (!origin[(width+2)*(_cur.y+1)+_cur.x+1] && origin[(width+2)*(_cur.y+1)+_cur.x]) {
        _next.y ++;
        openset[width * _cur.y + _cur.x - 1] = false;
        if (origin[(width+2)*_cur.y+_cur.x]) {
            openset[width * (_cur.y - 1) + _cur.x - 1] = false;
        }
    } else if (!origin[(width+2)*(_cur.y+1)+_cur.x] && origin[(width+2)*_cur.y+_cur.x]) {
        _next.x --;
        openset[width * (_cur.y - 1) + _cur.x - 1] = false;
        if (origin[(width+2)*_cur.y+_cur.x+1]) {
            openset[width * (_cur.y - 1) + _cur.x] = false;
        }
    }
    points.push_back(_next);
    // next steps
    for (;;) {
        _cur = points.back();
        _next = _cur;
        if (!origin[(width+2)*_cur.y+_cur.x]     && origin[(width+2)*_cur.y+_cur.x+1]
            && origin[(width+2)*(_cur.y+1)+_cur.x] && !origin[(width+2)*(_cur.y+1)+_cur.x+1]) {
            if (points[points.size()-2].x == _cur.x+1) {
                _next.y --;
                openset[width * (_cur.y - 1) + _cur.x] = false;
            } else {
                _next.y ++;
                openset[width * _cur.y + _cur.x - 1] = false;
            }
        } else if (origin[(width+2)*_cur.y+_cur.x]     && !origin[(width+2)*_cur.y+_cur.x+1]
                   && !origin[(width+2)*(_cur.y+1)+_cur.x] && origin[(width+2)*(_cur.y+1)+_cur.x+1]) {
            if (points[points.size()-2].y == _cur.y+1) {
                _next.x ++;
                openset[width * _next.y + _cur.x] = false;
            } else {
                _next.x --;
                openset[width * (_next.y - 1) + _cur.x - 1] = false;
            }
        } else if (!origin[(width+2)*_cur.y+_cur.x] && origin[(width+2)*_cur.y+_cur.x+1]) {
            _next.y --;
            openset[width * (_cur.y - 1) + _cur.x] = false;
            if (origin[(width+2)*(_cur.y+1)+_cur.x+1]) {
                openset[width * _cur.y + _cur.x] = false;
            }
        } else if (!origin[(width+2)*_cur.y+_cur.x+1] && origin[(width+2)*(_cur.y+1)+_cur.x+1]) {
            _next.x ++;
            openset[width * _cur.y + _cur.x] = false;
            if (origin[(width+2)*(_cur.y+1)+_cur.x]) {
                openset[width * _cur.y + _cur.x - 1] = false;
            }
        } else if (!origin[(width+2)*(_cur.y+1)+_cur.x+1] && origin[(width+2)*(_cur.y+1)+_cur.x]) {
            _next.y ++;
            openset[width * _cur.y + _cur.x - 1] = false;
            if (origin[(width+2)*_cur.y+_cur.x]) {
                openset[width * (_cur.y - 1) + _cur.x - 1] = false;
            }
        } else if (!origin[(width+2)*(_cur.y+1)+_cur.x] && origin[(width+2)*_cur.y+_cur.x]) {
            _next.x --;
            openset[width * (_cur.y - 1) + _cur.x - 1] = false;
            if (origin[(width+2)*_cur.y+_cur.x+1]) {
                openset[width * (_cur.y - 1) + _cur.x] = false;
            }
        }
        if (_next.x == points[0].x && _next.y == points[0].y) {
            break;
        }
        points.push_back(_next);
    }
    Ring ring;
    ring.nPoints = points.size();
    ring.points = (Point*)malloc(ring.nPoints*sizeof(Point));
    memcpy(ring.points, points.data(), ring.nPoints*sizeof(Point));
    return ring;
} // Ring traceEdge

std::vector<Ring> ExtractRings(int width, int height, const bool *origin, bool* edge) {
    bool* _openset = (bool*)malloc(width*height*sizeof(bool));
    memcpy(_openset, edge, width*height*sizeof(bool));
    std::vector<Ring> rings;
    for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
        if(!_openset[width*y+x]) continue;
        Ring _ring = traceEdge(width, height, x, y, origin, _openset);
        rings.push_back(_ring);
    } // for x
    } // for y
    free(_openset);
    return rings;
} // std::vector<Ring> ExtractRings

void simplifyDP(const Ring &origin, Ring& output, float maxDistThres) {
    int maxIdx; float maxDist=0;
    float dx0 = origin.points[origin.nPoints-1].x - origin.points[0].x;
    float dy0 = origin.points[origin.nPoints-1].y - origin.points[0].y;
    float dnorm0 = sqrt(dx0*dx0+dy0*dy0);
    for (int i = 1; i < origin.nPoints - 1; ++i) {
        float dist;
        float dx1 = origin.points[i].x - origin.points[0].x;
        float dy1 = origin.points[i].y - origin.points[0].y;
        if (dnorm0==0) {
            dist = sqrt(dx1*dx1+dy1*dy1);
        } else {
            dist = abs(dx0*dy1-dx1*dy0) / dnorm0;
        }
        if (dist > maxDist) {
            maxIdx = i;
            maxDist = dist;
        }
    }
    if (maxDist < maxDistThres) {
        output.nPoints = 2;
        output.points = (Point*)malloc(2*sizeof(Point));
        output.points[0] = origin.points[0];
        output.points[1] = origin.points[origin.nPoints-1];
        return;
    }
    Ring segments0, segments1;
    segments0.nPoints = maxIdx+1;
    segments0.points = (Point*)malloc(segments0.nPoints*sizeof(Point));
    memcpy(segments0.points, origin.points, segments0.nPoints*sizeof(Point));
    segments1.nPoints = origin.nPoints - maxIdx;
    segments1.points = (Point*)malloc(segments1.nPoints*sizeof(Point));
    memcpy(segments1.points, &origin.points[maxIdx], segments1.nPoints*sizeof(Point));
    Ring segments0Output, segments1Output;
    simplifyDP(segments0, segments0Output, maxDistThres);
    simplifyDP(segments1, segments1Output, maxDistThres);
    free(segments0.points);
    free(segments1.points);
    output.nPoints = segments0Output.nPoints + segments1Output.nPoints - 1;
    output.points = (Point*)malloc(output.nPoints*sizeof(Point));
    memcpy(output.points, segments0Output.points, (segments0Output.nPoints-1)*sizeof(Point));
    memcpy(&output.points[segments0Output.nPoints-1], segments1Output.points, segments1Output.nPoints*sizeof(Point));
    free(segments0Output.points);
    free(segments1Output.points);
}

void RingSimplifyDP(const Ring &origin, Ring &output, float maxDistThres) {
    if (origin.nPoints < 3) {
        throw std::runtime_error("[Pix2Vec]RingSimplifyDP: A ring must consisets of at least 3 points.");
    }
    Ring _input, _output;
    _input.nPoints = origin.nPoints+1;
    _input.points = (Point*)malloc(_input.nPoints*sizeof(Point));
    memcpy(_input.points, origin.points, origin.nPoints*sizeof(Point));
    _input.points[origin.nPoints] = origin.points[0];
    simplifyDP(_input, _output, maxDistThres);
    output.nPoints = _output.nPoints-1;
    output.points = (Point*)malloc(output.nPoints*sizeof(Point));
    memcpy(output.points, _output.points, output.nPoints*sizeof(Point));
    free(_input.points);
    free(_output.points);
} // RingSimplifyDP

bool IsClockwise(const Ring& ring) {
    int area = 0;
    for (int i = 0; i < ring.nPoints; ++i) {
        int j = (i + 1) % ring.nPoints;
        area += (ring.points[i].x * ring.points[j].y) - (ring.points[i].y * ring.points[j].x);
    }
    return area > 0;
} // bool IsClockwise

bool IsPointInRing(const Point& p, const Ring& ring) {
    bool inside = false;
    int n = ring.nPoints;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        auto& pi = ring.points[i];
        auto& pj = ring.points[j];
        if ((pi.y > p.y) != (pj.y > p.y) &&
            (p.x < (pj.x - pi.x) * (p.y - pi.y) / (pj.y - pi.y) + pi.x)) {
            inside = !inside;
        }
    }
    return inside;
} // bool IsPointInRing

void BuildMultiPolygon(const std::vector<Ring> &outerRings, const std::vector<Ring> &innerRings, MultiPolygon& output) {
    output.nPolygons = static_cast<int>(outerRings.size());
    output.polygons = (Polygon*)malloc(output.nPolygons*sizeof(Polygon));
    for (int i = 0; i < output.nPolygons; ++i) {
        output.polygons[i].nInnerRings = 0;
        output.polygons[i].outerRing = outerRings[i];
    }
    std::vector<std::vector<Ring>> _innerRings(output.nPolygons);
    for (auto& innerRing : innerRings) {
        std::vector<int> maybeOuterRingIdx;
        for (int i = 0; i < outerRings.size(); ++i) {
            if( Pix2Vec::IsPointInRing(innerRing.points[0], outerRings[i]) ) {
                maybeOuterRingIdx.push_back(i);
            }
        }
        if (maybeOuterRingIdx.empty()) {
            throw std::runtime_error("[Pix2Vec]BuildMultiPolygon: illegal inner ring without an outer ring.");
        } else if (maybeOuterRingIdx.size() == 1) {
            _innerRings[maybeOuterRingIdx[0]].push_back(innerRing);
            continue;
        }
        for (int i = 0; i < maybeOuterRingIdx.size(); ++i) {
            bool theChosen = true;
            const auto& _currentRing = outerRings[maybeOuterRingIdx[i]];
            for (int j = 0; j < maybeOuterRingIdx.size(); ++j) {
                if (i == j) continue;
                const auto& _anotherRing = outerRings[maybeOuterRingIdx[j]];
                if ( Pix2Vec::IsPointInRing(_anotherRing.points[0], _currentRing) ) {
                    theChosen = false;
                    break;
                }
            }
            if (theChosen) {
                _innerRings[maybeOuterRingIdx[i]].push_back(innerRing);
                break;
            }
        }
    }
    for (int i = 0; i < output.nPolygons; ++i) {
        int nInnerRings = static_cast<int>(_innerRings[i].size());
        Polygon& polygon = output.polygons[i];
        polygon.nInnerRings = nInnerRings;
        polygon.innerRing = (Ring*)malloc(nInnerRings*sizeof(Ring));
        memcpy(polygon.innerRing, _innerRings[i].data(), nInnerRings*sizeof(Ring));
        for (int j = 0; j < polygon.nInnerRings; ++j) {
            polygon.innerRing[j].points = (Point*)malloc(_innerRings[i][j].nPoints*sizeof(Point));
            memcpy(polygon.innerRing[j].points, _innerRings[i][j].points, _innerRings[i][j].nPoints*sizeof(Point));
        }
    }
} // void BuildMultiPolygon

void Vecterize(int width, int height, bool* input, MultiPolygon& output, float maxDistThres) {
    bool* paded_input = (bool*)malloc((width+2)*(height+2)*sizeof(bool));
    Padding(width, height, input, paded_input);
    bool* edge = (bool*)malloc(width*height*sizeof(bool));
    ExtractEdges(width, height, paded_input, edge);
    std::vector<Ring> rings = ExtractRings(width, height, paded_input, edge);
    std::vector<Ring> outerRings, innerRings;
    for (auto& ring : rings) {
        Ring _ringSimp;
        RingSimplifyDP(ring, _ringSimp, maxDistThres);
        if (_ringSimp.nPoints < 3) continue;
        if ( IsClockwise(_ringSimp) ) {
            outerRings.push_back(_ringSimp);
        } else {
            innerRings.push_back(_ringSimp);
        }
    }
    BuildMultiPolygon(outerRings, innerRings, output);
    // 释放资源
    free(edge);
    free(paded_input);
} // void Vecterize

void FreePolygon(Polygon& polygon) {
    for (int j = 0; j < polygon.nInnerRings; ++j) {
        free(polygon.innerRing[j].points);
    }
    free(polygon.innerRing);
} // void FreePolygon

void FreeMultiPolygon(MultiPolygon& multiPolygon) {
    for (int i = 0; i < multiPolygon.nPolygons; ++i) {
        FreePolygon(multiPolygon.polygons[i]);
    }
    free(multiPolygon.polygons);
} // void FreeMultiPolygon
} // namespace Pix2Vec