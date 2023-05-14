#version 460 core

layout (early_fragment_tests) in;

layout (location = 0) in flat uint i_object_id;

layout (std430, binding = 5) writeonly restrict buffer b_roc_output {
    uint[] roc_visibility;
};

void main() {
    roc_visibility[i_object_id] = 1;
}
