//Copyright (c) [2024] [Initial_Equationor]
//[pix2vec] is licensed under Mulan PubL v2.
//You can use this software according to the terms and conditions of the Mulan PubL v2.
//You may obtain a copy of Mulan PubL v2 at:
//http://license.coscl.org.cn/MulanPubL-2.0
//THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
//EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
//MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//See the Mulan PubL v2 for more details.
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "stb_image.h"
#include "pix2vec.hpp"

void printMultiPolygon(Pix2Vec::MultiPolygon& multiPolygon){
    for (int i = 0; i < multiPolygon.nPolygons; ++i) {
        std::cout << "polygon " << i << ":" << std::endl;
        auto polygon = multiPolygon.polygons[i];
        std::cout << " outerRing:";
        for (int j = 0; j < polygon.outerRing.nPoints; ++j) {
            std::cout << "(" << polygon.outerRing.points[j].x << "," << polygon.outerRing.points[j].y << ") ";
        }
        std::cout << std::endl;
        for (int j = 0; j < polygon.nInnerRings; ++j) {
            std::cout << " innerRing " << j << ":";
            for (int k = 0; k < polygon.innerRing[j].nPoints; ++k) {
                std::cout << "(" << polygon.innerRing[j].points[k].x << "," << polygon.innerRing[j].points[k].y << ") ";
            }
            std::cout << std::endl;
        }
    }
} // void printMultiPolygon

TEST(pix2vec, ExtractEdges) {
    bool input[16] = {0, 0, 1, 0,
                      0, 1, 1, 1,
                      0, 1, 1, 1,
                      0, 1, 1, 1 };
    bool paded_input[36];
    Pix2Vec::Padding(4, 4, input, paded_input);
    bool output[16];
    Pix2Vec::ExtractEdges(4, 4, paded_input, output);
    bool test_edge[16] = { 0, 0, 1, 0,
                           0, 1, 1, 1,
                           0, 1, 0, 1,
                           0, 1, 1, 1 };
    for (int i = 0; i < 16; ++i) {
        ASSERT_EQ(output[i], test_edge[i]) << "x=" << i%4 << ", y=" << i/4;
    }
}

TEST(pix2vec, ExtractRings) {
    bool input[16] = { 0, 0, 0, 0,
                       0, 1, 1, 1,
                       0, 1, 1, 1,
                       0, 0, 0, 0 };
    bool paded_input[36];
    Pix2Vec::Padding(4, 4, input, paded_input);
    bool edge[16];
    try {
        Pix2Vec::ExtractEdges(4, 4, paded_input, edge);
    } catch(std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
    }
    bool test_edge[16] = { 0, 0, 0, 0,
                           0, 1, 1, 1,
                           0, 1, 1, 1,
                           0, 0, 0, 0 };
    for (int i = 0; i < 16; ++i) {
        ASSERT_EQ(edge[i], test_edge[i]) << "x=" << i%4 << ", y=" << i/4;
    }
    std::vector<Pix2Vec::Ring> rings = Pix2Vec::ExtractRings(4, 4, paded_input, edge);
    Pix2Vec::Ring ring = rings[0];
    ASSERT_EQ(ring.nPoints, 10) << "incorrect points number";
    int idx=0, x=1, y=1;
    ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";
    for (int i = 0; i < 3; ++i) {
        idx++; x++;
        ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";
    }
    for (int i = 0; i < 2; ++i) {
        idx++; y++;
        ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";
    }
    for (int i = 0; i < 3; ++i) {
        idx++; x--;
        ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";
    }
    idx++; y--;
    ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";
//    for (int i=0;i<rings.size();i++) {
//        std::cout << "ring:" << i << std::endl;
//        for (int j = 0; j < rings[i].nPoints; ++j) {
//            std::cout << "(" << rings[i].points[j].x << ","
//                << rings[i].points[j].y << "),";
//        }
//        std::cout << std::endl;
//    }
}

TEST(pix2vec, ExtractRings2) {
    bool input[16] = { 1, 0, 0, 0,
                       0, 1, 0, 0,
                       0, 1, 0, 1,
                       0, 0, 0, 0 };
    bool paded_input[36];
    Pix2Vec::Padding(4, 4, input, paded_input);
    bool edge[16];
    Pix2Vec::ExtractEdges(4, 4, paded_input, edge);
    bool test_edge[16] = { 1, 0, 0, 0,
                           0, 1, 0, 0,
                           0, 1, 0, 1,
                           0, 0, 0, 0 };
    for (int i = 0; i < 16; ++i) {
        ASSERT_EQ(edge[i], test_edge[i]) << "x=" << i%4 << ", y=" << i/4;
    }
    std::vector<Pix2Vec::Ring> rings = Pix2Vec::ExtractRings(4, 4, paded_input, edge);
    ASSERT_EQ(rings.size(), 3) << "incorrect ring number";
    Pix2Vec::Ring ring = rings[0];
    ASSERT_EQ(ring.nPoints, 4) << "ring 0 incorrect points number";
    int idx=0, x=0, y=0;
    ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";
    idx++; x++;
    ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";
    idx++; y++;
    ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";
    idx++; x--;
    ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";

    ring = rings[1];
    ASSERT_EQ(ring.nPoints, 6) << "ring 1 incorrect points number";
    idx=0, x=1, y=1;
    idx++; x++;
    ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";
    idx++; y++;
    ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";
    idx++; y++;
    ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";
    idx++; x--;
    ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";
    idx++; y--;
    ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";

    ring = rings[2];
    ASSERT_EQ(ring.nPoints, 4) << "ring 2 incorrect points number";
    idx=0, x=3, y=2;
    ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";
    idx++; x++;
    ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";
    idx++; y++;
    ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";
    idx++; x--;
    ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";
//    for (int i=0;i<rings.size();i++) {
//        std::cout << "ring:" << i << std::endl;
//        for (int j = 0; j < rings[i].nPoints; ++j) {
//            std::cout << "(" << rings[i].points[j].x << ","
//                << rings[i].points[j].y << "),";
//        }
//        std::cout << std::endl;
//    }
}

TEST(pix2vec, ExtractRings3) {
    bool input[16] = { 1, 1, 1, 1,
                       1, 1, 1, 1,
                       1, 1, 0, 1,
                       1, 1, 1, 1 };
    bool paded_input[36];
    Pix2Vec::Padding(4, 4, input, paded_input);
    bool edge[16];
    Pix2Vec::ExtractEdges(4, 4, paded_input, edge);
    bool test_edge[16] = { 1, 1, 1, 1,
                           1, 1, 1, 1,
                           1, 1, 0, 1,
                           1, 1, 1, 1 };
    for (int i = 0; i < 16; ++i) {
        ASSERT_EQ(edge[i], test_edge[i]) << "x=" << i%4 << ", y=" << i/4;
    }
    std::vector<Pix2Vec::Ring> rings = Pix2Vec::ExtractRings(4, 4, paded_input, edge);
    ASSERT_EQ(rings.size(), 2) << "incorrect ring number";
    Pix2Vec::Ring ring = rings[0];
    ASSERT_EQ(ring.nPoints, 16) << "ring 0 incorrect points number";
    int idx=0, x=0, y=0;
    for (int i = 0; i < 4; ++i) {
        idx++; x++;
        ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";
    }
    for (int i = 0; i < 4; ++i) {
        idx++; y++;
        ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";
    }
    for (int i = 0; i < 4; ++i) {
        idx++; x--;
        ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";
    }
    for (int i = 0; i < 3; ++i) {
        idx++; y--;
        ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";
    }

    ring = rings[1];
    ASSERT_EQ(ring.nPoints, 4) << "ring 1 incorrect points number";
    idx=0, x=2, y=2;
    idx++; y++;
    ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";
    idx++; x++;
    ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";
    idx++; y--;
    ASSERT_EQ(ring.points[idx].x, x) << "point incorrect"; ASSERT_EQ(ring.points[idx].y, y) << "point incorrect";

//    for (int i=0;i<rings.size();i++) {
//        std::cout << "ring:" << i << std::endl;
//        for (int j = 0; j < rings[i].nPoints; ++j) {
//            std::cout << "(" << rings[i].points[j].x << ","
//                << rings[i].points[j].y << "),";
//        }
//        std::cout << std::endl;
//    }
}

TEST(pix2vec, RingSimplifyDP) {
    Pix2Vec::Ring origin, output;
    origin.nPoints = 4;
    origin.points = (Pix2Vec::Point*)malloc(4*sizeof(Pix2Vec::Point));
    origin.points[0] = Pix2Vec::Point{0,0};
    origin.points[1] = Pix2Vec::Point{1,0};
    origin.points[2] = Pix2Vec::Point{1,1};
    origin.points[3] = Pix2Vec::Point{0,1};
    Pix2Vec::RingSimplifyDP(origin, output);
    ASSERT_EQ(output.nPoints, 1) << "not simplify correctly";
    free(origin.points);
    free(output.points);
    // 正常抽稀的情况
    origin.nPoints = 16;
    origin.points = (Pix2Vec::Point*)malloc(16*sizeof(Pix2Vec::Point));
    int idx=0, x=0, y=0;
    origin.points[idx] = Pix2Vec::Point{x,y};
    idx ++;
    for (int i = 0; i < 4; ++i) {
        x++;
        origin.points[idx] = Pix2Vec::Point{x,y};
        idx ++;
    }
    for (int i = 0; i < 4; ++i) {
        y++;
        origin.points[idx] = Pix2Vec::Point{x,y};
        idx ++;
    }
    for (int i = 0; i < 4; ++i) {
        x--;
        origin.points[idx] = Pix2Vec::Point{x,y};
        idx ++;
    }
    for (int i = 0; i < 3; ++i) {
        y--;
        origin.points[idx] = Pix2Vec::Point{x,y};
        idx ++;
    }
    Pix2Vec::RingSimplifyDP(origin, output);
    ASSERT_EQ(output.nPoints, 4) << "not simplify correctly";
    free(origin.points);
    free(output.points);
}

TEST(pix2vec, IsClockwise) {
    Pix2Vec::Ring ring;
    ring.nPoints = 4;
    ring.points = (Pix2Vec::Point*)malloc(4*sizeof(Pix2Vec::Point));
    ring.points[0] = Pix2Vec::Point{ 0, 0 };
    ring.points[1] = Pix2Vec::Point{ 4, 0 };
    ring.points[2] = Pix2Vec::Point{ 4, 4 };
    ring.points[3] = Pix2Vec::Point{ 0, 4 };
    bool isClockwise = Pix2Vec::IsClockwise(ring);
    ASSERT_TRUE(isClockwise) << "ring is clockwise";
    ring.points[0] = Pix2Vec::Point{ 0, 0 };
    ring.points[1] = Pix2Vec::Point{ 0, 4 };
    ring.points[2] = Pix2Vec::Point{ 4, 4 };
    ring.points[3] = Pix2Vec::Point{ 4, 0 };
    isClockwise = Pix2Vec::IsClockwise(ring);
    ASSERT_FALSE(isClockwise) << "ring is anticlockwise";
    free(ring.points);
}

TEST(pix2vec, IsPointInRing) {
    Pix2Vec::Ring ring;
    ring.nPoints = 4;
    ring.points = (Pix2Vec::Point*)malloc(4*sizeof(Pix2Vec::Point));
    ring.points[0] = Pix2Vec::Point{ 1, 1 };
    ring.points[1] = Pix2Vec::Point{ 3, 1 };
    ring.points[2] = Pix2Vec::Point{ 3, 3 };
    ring.points[3] = Pix2Vec::Point{ 1, 3 };
    bool test_result[25] = { 0, 0, 0, 0, 0,
                             0, 1, 1, 0, 0,
                             0, 1, 1, 0, 0,
                             0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0 };
    for (int y = 0; y < 5; ++y) {
    for (int x = 0; x < 5; ++x) {
        auto p = Pix2Vec::Point{ x, y };
        ASSERT_EQ(test_result[5*y+x], Pix2Vec::IsPointInRing(p, ring))
            << "(" << x << "," << y << ")";
    } // for x
    } // for y
    free(ring.points);
}

TEST(pix2vec, RingContain) {
    Pix2Vec::Ring outerRing;
    outerRing.nPoints = 4;
    outerRing.points = (Pix2Vec::Point*)malloc(4 * sizeof(Pix2Vec::Point));
    outerRing.points[0] = Pix2Vec::Point{1, 1 };
    outerRing.points[1] = Pix2Vec::Point{4, 1 };
    outerRing.points[2] = Pix2Vec::Point{4, 4 };
    outerRing.points[3] = Pix2Vec::Point{1, 4 };
    Pix2Vec::Ring innerRing;
    innerRing.nPoints = 4;
    innerRing.points = (Pix2Vec::Point*)malloc(4 * sizeof(Pix2Vec::Point));
    innerRing.points[0] = Pix2Vec::Point{2, 2 };
    innerRing.points[1] = Pix2Vec::Point{2, 3 };
    innerRing.points[2] = Pix2Vec::Point{3, 3 };
    innerRing.points[3] = Pix2Vec::Point{3, 2 };
    ASSERT_TRUE(Pix2Vec::IsPointInRing(innerRing.points[0], outerRing));
    ASSERT_TRUE(Pix2Vec::IsPointInRing(innerRing.points[1], outerRing));
    ASSERT_TRUE(Pix2Vec::IsPointInRing(innerRing.points[2], outerRing));
    ASSERT_TRUE(Pix2Vec::IsPointInRing(innerRing.points[3], outerRing));
    free(outerRing.points);
    free(innerRing.points);
}

TEST(pix2vec, BuildMultiPolygon) {
    std::vector<Pix2Vec::Ring> outerRings, innerRings;
    Pix2Vec::Ring outerRing1;
    outerRing1.nPoints = 4;
    outerRing1.points = new Pix2Vec::Point[4]{ {0,0}, {1,0}, {1,1}, {0,1} };
    outerRings.push_back(outerRing1);
    Pix2Vec::Ring outerRing2;
    outerRing2.nPoints = 4;
    outerRing2.points = new Pix2Vec::Point[4]{ {1,1}, {4,1}, {4,4}, {1,4} };
    outerRings.push_back(outerRing2);
    Pix2Vec::Ring innerRing1;
    innerRing1.nPoints = 4;
    innerRing1.points = new Pix2Vec::Point[4]{ {2,2}, {2,3}, {3,3}, {3,2} };
    innerRings.push_back(innerRing1);
    Pix2Vec::MultiPolygon multiPolygon;
    Pix2Vec::BuildMultiPolygon(outerRings, innerRings, multiPolygon);
    ASSERT_EQ(multiPolygon.nPolygons, 2);
    ASSERT_EQ(multiPolygon.polygons[0].nInnerRings, 0);
    ASSERT_EQ(multiPolygon.polygons[1].nInnerRings, 1);
    // 释放资源
    for (int i = 0; i < multiPolygon.nPolygons; ++i) {
        free(multiPolygon.polygons[i].innerRing);
    }
    free(multiPolygon.polygons);
    delete outerRing1.points;
    delete outerRing2.points;
    delete innerRing1.points;
    // TODO: 内环中镶嵌外环的情况
    outerRings.clear(); innerRings.clear();
    outerRing1.nPoints = 4;
    outerRing1.points = new Pix2Vec::Point[4]{ {0,0}, {5,0}, {5,5}, {0,5} };
    outerRings.push_back(outerRing1);
    outerRing2.nPoints = 4;
    outerRing2.points = new Pix2Vec::Point[4]{ {2,2}, {3,2}, {3,3}, {2,3} };
    outerRings.push_back(outerRing2);
    innerRing1.nPoints = 4;
    innerRing1.points = new Pix2Vec::Point[4]{ {1,1}, {1,4}, {4,4}, {4,1} };
    innerRings.push_back(innerRing1);
    Pix2Vec::BuildMultiPolygon(outerRings, innerRings, multiPolygon);
    ASSERT_EQ(multiPolygon.nPolygons, 2);
    ASSERT_EQ(multiPolygon.polygons[0].nInnerRings, 1);
    ASSERT_EQ(multiPolygon.polygons[1].nInnerRings, 0);
    // 释放资源
    Pix2Vec::FreeMultiPolygon(multiPolygon);
    delete outerRing1.points;
    delete outerRing2.points;
    delete innerRing1.points;
}

TEST(pix2vec, Vecterize) {
    bool input[36] = {1, 0, 1, 0, 0, 0,
                      0, 1, 1, 1, 1, 1,
                      0, 1, 0, 0, 0, 1,
                      1, 1, 0, 1, 0, 1,
                      1, 1, 0, 0, 0, 1,
                      1, 1, 1, 1, 1, 1 };
    Pix2Vec::MultiPolygon multiPolygon;
    Pix2Vec::Vecterize(6, 6, input, multiPolygon, 0.5f);
//    printMultiPolygon(multiPolygon);
    ASSERT_EQ(multiPolygon.nPolygons, 3);
    ASSERT_EQ(multiPolygon.polygons[0].nInnerRings, 0);
    ASSERT_EQ(multiPolygon.polygons[1].nInnerRings, 1);
    ASSERT_EQ(multiPolygon.polygons[2].nInnerRings, 0);
    Pix2Vec::FreeMultiPolygon(multiPolygon);
}

TEST(pix2vec, Png) {
    int width, height, channels;
    unsigned char* img = stbi_load("assets/test.png", &width, &height, &channels, 0);
    ASSERT_EQ(width,128);
    ASSERT_EQ(height,128);
    ASSERT_EQ(channels,3);
    bool input[128*128];
    for (int i = 0; i < 128*128; ++i) {
        input[i] = img[3*i];
    }
    Pix2Vec::MultiPolygon multiPolygon;
    Pix2Vec::Vecterize(128, 128, input, multiPolygon);
//    printMultiPolygon(multiPolygon);
    ASSERT_EQ(multiPolygon.nPolygons, 5);
    ASSERT_EQ(multiPolygon.polygons[0].nInnerRings, 0);
    ASSERT_EQ(multiPolygon.polygons[1].nInnerRings, 1);
    ASSERT_EQ(multiPolygon.polygons[2].nInnerRings, 1);
    ASSERT_EQ(multiPolygon.polygons[3].nInnerRings, 0);
    ASSERT_EQ(multiPolygon.polygons[4].nInnerRings, 0);
    Pix2Vec::FreeMultiPolygon(multiPolygon);
    stbi_image_free(img);
}

//int main() {
//    testing::InitGoogleTest();
//    return RUN_ALL_TESTS();
//}
