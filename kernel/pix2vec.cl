// Copyright (c) [2024] [Initial_Equationor]
// [pix2vec] is licensed under Mulan PubL v2.
// You can use this software according to the terms and conditions of the Mulan PubL v2.
// You may obtain a copy of Mulan PubL v2 at:
//          http://license.coscl.org.cn/MulanPubL-2.0
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PubL v2 for more details.
__kernel void extract_edges(
    const int width,
    const int height,
    __global const bool *input, // paded images
    __global bool* output
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (!input[(width+2)*(y+1)+x+1]) {
        output[width*y+x] = false;
        return;
    }
    bool notEdge = input[(width+2)*y+x]     && input[(width+2)*y+x+1]     && input[(width+2)*y+x+2]
                 && input[(width+2)*(y+1)+x] && input[(width+2)*(y+1)+x+1] && input[(width+2)*(y+1)+x+2]
                 && input[(width+2)*(y+2)+x] && input[(width+2)*(y+2)+x+1] && input[(width+2)*(y+2)+x+2];
    output[width*y+x] = !notEdge;
}