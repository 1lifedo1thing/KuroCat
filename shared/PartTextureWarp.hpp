// Software warpers and geometry helpers for KuroCat 1.6.0 `partTextures`, shared by Windows (GDI+) and Linux (Cairo).
// All canonical part-space UV bounding boxes match KuroCat-Workshop/spec.js and Sources/CatRenderer.swift identically.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "Rig.hpp"

namespace partwarp {

struct UVBox {
    float x0, y0, x1, y1;
};

// Canonical part-space bounding boxes (must stay identical across Web, macOS, Windows, Linux)
constexpr UVBox HEAD_UV       = { -76.0f,  -66.0f,  76.0f,  66.0f };
constexpr UVBox BODY_UV       = { -72.0f, -132.0f,  72.0f,  24.0f };
constexpr UVBox EAR_UV        = { -75.0f, -125.0f,  75.0f,  25.0f };
constexpr UVBox LIMB_UV       = {  -8.0f,  -14.0f, 122.0f,  14.0f };
constexpr UVBox TAIL_UV       = { -10.0f,  -10.0f, 114.0f,  10.0f };
constexpr UVBox IRIS_UV       = { -14.0f,  -13.0f,  14.0f,  13.0f };
constexpr UVBox HIND_PAW_UV   = {  -1.1f,   -1.1f,   1.1f,   1.1f };

// Head dome warp output grid (covers [-64, +64] x [-58, +58] at 2x scale = 256 x 232)
constexpr int HEAD_WARP_W = 256;
constexpr int HEAD_WARP_H = 232;
constexpr float HEAD_WARP_X0 = -64.0f;
constexpr float HEAD_WARP_Y0 = -58.0f;
constexpr float HEAD_WARP_X1 =  64.0f;
constexpr float HEAD_WARP_Y1 =  58.0f;

struct Vec2 { float x, y; };

struct CubicBezier {
    Vec2 p0, p1, p2, p3;
};

struct StripSample {
    Vec2 left;
    Vec2 right;
    float xCoord; // part-space x (e.g. -8..122 for limbs, -10..114 for tail)
};

// De Casteljau split of cubic Bezier at t in [0, 1]
inline std::pair<CubicBezier, CubicBezier> splitCubicBezier(Vec2 p0, Vec2 p1, Vec2 p2, Vec2 p3, float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    auto lerpPt = [&](Vec2 a, Vec2 b) -> Vec2 { return { a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t }; };
    Vec2 q0 = lerpPt(p0, p1);
    Vec2 q1 = lerpPt(p1, p2);
    Vec2 q2 = lerpPt(p2, p3);
    Vec2 r0 = lerpPt(q0, q1);
    Vec2 r1 = lerpPt(q1, q2);
    Vec2 s  = lerpPt(r0, r1);
    return { { p0, q0, r0, s }, { s, r1, q2, p3 } };
}

// Extract sub-curve [tStart, tEnd] of cubic Bezier [0, 1] via two De Casteljau splits
inline CubicBezier subCubicBezier(Vec2 p0, Vec2 p1, Vec2 p2, Vec2 p3, float tStart, float tEnd) {
    tStart = std::clamp(tStart, 0.0f, 1.0f);
    tEnd   = std::clamp(tEnd, tStart, 1.0f);
    CubicBezier seg = (tEnd < 0.9999f)
        ? splitCubicBezier(p0, p1, p2, p3, tEnd).first
        : CubicBezier{ p0, p1, p2, p3 };
    if (tStart > 0.001f && tEnd > 1e-5f) {
        return splitCubicBezier(seg.p0, seg.p1, seg.p2, seg.p3, tStart / tEnd).second;
    }
    return seg;
}

// Find parameter t where tail curve crosses outside body hip width so tail root stays tucked behind torso
// even when the tail sweeps in front of the body (sleeping curl / 4-key tail typing). Matches renderer.js.
inline float findTailFrontSplitT(Vec2 root, Vec2 c1, Vec2 c2, Vec2 tip, float seatX, float bodyWidth, float lieIntensity) {
    float lie = std::clamp(lieIntensity, 0.0f, 1.0f);
    float lieS = lie * lie * (3.0f - 2.0f * lie);
    float hipRightX = seatX + 49.0f * bodyWidth * (1.0f + lieS * kurorig::LieWiden);
    float prevDx = root.x - hipRightX;
    for (int i = 1; i <= 24; ++i) {
        float t = (float)i / 24.0f;
        float u = 1.0f - t;
        float x = u*u*u*root.x + 3*u*u*t*c1.x + 3*u*t*t*c2.x + t*t*t*tip.x;
        float dx = x - hipRightX;
        if (dx < 0.0f && prevDx >= 0.0f) return std::max(0.0f, t - 0.04f);
        if (dx > 12.0f && i >= 4) return t;
        prevDx = dx;
    }
    return 0.28f;
}

// Bilinear sample from premultiplied RGBA8888 buffer (r, g, b, a in byte order) at normalized u,v in [0,1].
// Writes premultiplied (r, g, b, a) floats in 0..255 range. Returns false if outside [0,1] bounds or zero alpha.
inline bool samplePremulRGBA(const uint8_t* src, int w, int h, float u, float v,
                               float& outR, float& outG, float& outB, float& outA) {
    if (!src || w <= 0 || h <= 0 || u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) return false;
    float sx = u * (float)(w - 1);
    float sy = v * (float)(h - 1);
    int x0 = (int)std::floor(sx);
    int y0 = (int)std::floor(sy);
    int x1 = std::min(w - 1, x0 + 1);
    int y1 = std::min(h - 1, y0 + 1);
    float fx = sx - (float)x0;
    float fy = sy - (float)y0;
    float w00 = (1.0f - fx) * (1.0f - fy);
    float w10 = fx * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    const uint8_t* p00 = src + (y0 * w + x0) * 4;
    const uint8_t* p10 = src + (y0 * w + x1) * 4;
    const uint8_t* p01 = src + (y1 * w + x0) * 4;
    const uint8_t* p11 = src + (y1 * w + x1) * 4;

    outA = p00[3] * w00 + p10[3] * w10 + p01[3] * w01 + p11[3] * w11;
    if (outA < 0.5f) return false;
    outR = p00[0] * w00 + p10[0] * w10 + p01[0] * w01 + p11[0] * w11;
    outG = p00[1] * w00 + p10[1] * w10 + p01[1] * w01 + p11[1] * w11;
    outB = p00[2] * w00 + p10[2] * w10 + p01[2] * w01 + p11[2] * w11;
    return true;
}

inline void writePixel(uint8_t* dst, float r, float g, float b, float a, bool bgraOrder) {
    uint8_t ia = (uint8_t)std::clamp((int)(a + 0.5f), 0, 255);
    uint8_t ir = (uint8_t)std::clamp((int)(r + 0.5f), 0, (int)ia);
    uint8_t ig = (uint8_t)std::clamp((int)(g + 0.5f), 0, (int)ia);
    uint8_t ib = (uint8_t)std::clamp((int)(b + 0.5f), 0, (int)ia);
    if (bgraOrder) {
        dst[0] = ib; dst[1] = ig; dst[2] = ir; dst[3] = ia;
    } else {
        dst[0] = ir; dst[1] = ig; dst[2] = ib; dst[3] = ia;
    }
}

// Analytic inverse projection of HEAD_UV into HEAD_WARP_W x HEAD_WARP_H (covers HEAD_WARP_X0..X1 x Y0..Y1).
// Exact inverse of kurorig::projectHead3D(..., headDepth(x, y, dome)).
// Zero triangle seam artifacts! `dstStrideBytes` >= HEAD_WARP_W * 4.
inline void warpHeadDomePixels(const uint8_t* srcRGBA, int srcW, int srcH,
                               bool dome, float yawRad, float pitchRad,
                               uint8_t* dst, int dstStrideBytes, bool bgraOrder) {
    if (!srcRGBA || !dst || srcW <= 0 || srcH <= 0) return;

    const float cy = std::cos(yawRad), sy = std::sin(yawRad);
    const float cp = std::cos(pitchRad), sp = std::sin(pitchRad);

    // Rotation matrix R = Rx(pitch) * Ry(yaw) (row-major):
    // Row 0: (cy,       0,  sy)
    // Row 1: (-sy*sp,  cp,  cy*sp)
    // Row 2: (-sy*cp, -sp,  cy*cp)
    // Direction d = R^T * (0,0,1)^T = (-sy*cp, -sp, cy*cp)
    const float dx = -sy * cp;
    const float dy = -sp;
    const float dz = cy * cp;
    const float invDz = (std::fabs(dz) > 1e-4f) ? (1.0f / dz) : 1e4f;

    const float invA2 = 1.0f / (kurorig::DomeRadiusX * kurorig::DomeRadiusX);
    const float invB2 = 1.0f / (kurorig::DomeRadiusY * kurorig::DomeRadiusY);
    const float invC2 = 1.0f / (kurorig::DomeDepth   * kurorig::DomeDepth);
    const float quadA = dx * dx * invA2 + dy * dy * invB2 + dz * dz * invC2;
    const float invQuadA = 1.0f / std::max(1e-6f, quadA);

    const float invU = 1.0f / (HEAD_UV.x1 - HEAD_UV.x0);
    const float invV = 1.0f / (HEAD_UV.y1 - HEAD_UV.y0);
    const float invCy = (std::fabs(cy) > 1e-4f) ? (1.0f / cy) : 1.0f;
    const float invCp = (std::fabs(cp) > 1e-4f) ? (1.0f / cp) : 1.0f;

    for (int pyIdx = 0; pyIdx < HEAD_WARP_H; ++pyIdx) {
        uint8_t* row = dst + pyIdx * dstStrideBytes;
        const float py = HEAD_WARP_Y0 + ((float)pyIdx + 0.5f) / (float)HEAD_WARP_H * (HEAD_WARP_Y1 - HEAD_WARP_Y0);
        const float oBaseY = py * cp;
        const float oBaseX = -py * sy * sp;
        const float oBaseZ = py * cy * sp;

        for (int pxIdx = 0; pxIdx < HEAD_WARP_W; ++pxIdx) {
            const float px = HEAD_WARP_X0 + ((float)pxIdx + 0.5f) / (float)HEAD_WARP_W * (HEAD_WARP_X1 - HEAD_WARP_X0);
            float hx, hy;
            if (!dome) {
                const float z = kurorig::PlaneDepth;
                hx = (px - z * sy) * invCy;
                hy = (py - (-hx * sy + z * cy) * sp) * invCp;
            } else {
                const float ox = px * cy + oBaseX;
                const float oy = oBaseY;
                const float oz = px * sy + oBaseZ;
                const float quadB = ox * dx * invA2 + oy * dy * invB2 + oz * dz * invC2;
                const float quadC = ox * ox * invA2 + oy * oy * invB2 + oz * oz * invC2 - 1.0f;
                const float disc = quadB * quadB - quadA * quadC;
                bool hitDome = false;
                if (disc >= 0.0f) {
                    const float pz = (-quadB + std::sqrt(disc)) * invQuadA;
                    const float z = oz + pz * dz;
                    if (z >= 0.0f) {
                        hx = ox + pz * dx;
                        hy = oy + pz * dy;
                        hitDome = true;
                    }
                }
                if (!hitDome) {
                    const float pz = -oz * invDz;
                    hx = ox + pz * dx;
                    hy = oy + pz * dy;
                }
            }
            const float u = (hx - HEAD_UV.x0) * invU;
            const float v = (hy - HEAD_UV.y0) * invV;
            float r, g, b, a;
            if (samplePremulRGBA(srcRGBA, srcW, srcH, u, v, r, g, b, a)) {
                writePixel(row + pxIdx * 4, r, g, b, a, bgraOrder);
            } else {
                uint32_t zero = 0;
                std::memcpy(row + pxIdx * 4, &zero, 4);
            }
        }
    }
}

// Rasterize a single triangle in bitmap coords ((dstW x dstH) mapped onto screen [minX, minX+logW] x [minY, minY+logH])
inline void rasterizeQuadTri(uint8_t* dst, int dstW, int dstH, int dstStrideBytes, bool bgraOrder,
                             const uint8_t* srcRGBA, int srcW, int srcH,
                             Vec2 p0, float u0, float v0,
                             Vec2 p1, float u1, float v1,
                             Vec2 p2, float u2, float v2) {
    float area = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
    if (std::fabs(area) < 1e-4f) return;
    if (area < 0.0f) {
        std::swap(p1, p2); std::swap(u1, u2); std::swap(v1, v2);
        area = -area;
    }
    const float invArea = 1.0f / area;

    int minPx = std::max(0, (int)std::floor(std::min({ p0.x, p1.x, p2.x }) - 0.5f));
    int maxPx = std::min(dstW - 1, (int)std::ceil(std::max({ p0.x, p1.x, p2.x }) + 0.5f));
    int minPy = std::max(0, (int)std::floor(std::min({ p0.y, p1.y, p2.y }) - 0.5f));
    int maxPy = std::min(dstH - 1, (int)std::ceil(std::max({ p0.y, p1.y, p2.y }) + 0.5f));

    for (int py = minPy; py <= maxPy; ++py) {
        float y = (float)py + 0.5f;
        uint8_t* row = dst + py * dstStrideBytes;
        for (int px = minPx; px <= maxPx; ++px) {
            float x = (float)px + 0.5f;
            float w0 = ((p1.x - x) * (p2.y - y) - (p2.x - x) * (p1.y - y)) * invArea;
            float w1 = ((p2.x - x) * (p0.y - y) - (p0.x - x) * (p2.y - y)) * invArea;
            float w2 = 1.0f - w0 - w1;
            if (w0 < -1e-4f || w1 < -1e-4f || w2 < -1e-4f) continue;
            float u = std::clamp(w0 * u0 + w1 * u1 + w2 * u2, 0.0f, 1.0f);
            float v = std::clamp(w0 * v0 + w1 * v1 + w2 * v2, 0.0f, 1.0f);
            float r, g, b, a;
            if (samplePremulRGBA(srcRGBA, srcW, srcH, u, v, r, g, b, a)) {
                writePixel(row + px * 4, r, g, b, a, bgraOrder);
            }
        }
    }
}

struct StripMeshResult {
    float minX = 0, minY = 0, width = 0, height = 0; // logical scene bounds
    int bmpW = 0, bmpH = 0;                          // rasterized bitmap pixel size
    bool valid = false;
};

// Rasterizes a curved strip defined by `samples` (left/right edges along a curve, with part-space `xCoord`)
// into `dstBuffer` (2x logical resolution, zero-initialized internally). Seamless across all triangles!
inline StripMeshResult rasterizeStripMeshPixels(const uint8_t* srcRGBA, int srcW, int srcH,
                                                const StripSample* samples, int sampleCount,
                                                UVBox uvBox, float scale2x,
                                                std::vector<uint8_t>& dstBuffer, bool bgraOrder,
                                                bool vertical = false) {
    StripMeshResult res;
    if (!srcRGBA || srcW <= 0 || srcH <= 0 || !samples || sampleCount < 2) return res;

    float minX = 1e9f, minY = 1e9f, maxX = -1e9f, maxY = -1e9f;
    for (int i = 0; i < sampleCount; ++i) {
        minX = std::min({ minX, samples[i].left.x, samples[i].right.x });
        minY = std::min({ minY, samples[i].left.y, samples[i].right.y });
        maxX = std::max({ maxX, samples[i].left.x, samples[i].right.x });
        maxY = std::max({ maxY, samples[i].left.y, samples[i].right.y });
    }
    minX -= 2.0f; minY -= 2.0f; maxX += 2.0f; maxY += 2.0f;
    float logW = maxX - minX, logH = maxY - minY;
    if (logW <= 0.5f || logH <= 0.5f) return res;

    int bmpW = std::clamp((int)std::ceil(logW * scale2x), 2, 1024);
    int bmpH = std::clamp((int)std::ceil(logH * scale2x), 2, 1024);
    float sx = (float)bmpW / logW;
    float sy = (float)bmpH / logH;

    dstBuffer.assign((size_t)bmpW * (size_t)bmpH * 4, 0);
    int stride = bmpW * 4;
    const float origin = vertical ? uvBox.y0 : uvBox.x0;
    const float span   = vertical ? (uvBox.y1 - uvBox.y0) : (uvBox.x1 - uvBox.x0);
    const float invSpan = 1.0f / std::max(1e-5f, span);

    for (int i = 0; i < sampleCount - 1; ++i) {
        const auto& s0 = samples[i];
        const auto& s1 = samples[i + 1];
        float t0 = (s0.xCoord - origin) * invSpan;
        float t1 = (s1.xCoord - origin) * invSpan;
        Vec2 l0 = { (s0.left.x  - minX) * sx, (s0.left.y  - minY) * sy };
        Vec2 r0 = { (s0.right.x - minX) * sx, (s0.right.y - minY) * sy };
        Vec2 l1 = { (s1.left.x  - minX) * sx, (s1.left.y  - minY) * sy };
        Vec2 r1 = { (s1.right.x - minX) * sx, (s1.right.y - minY) * sy };

        if (vertical) {
            rasterizeQuadTri(dstBuffer.data(), bmpW, bmpH, stride, bgraOrder, srcRGBA, srcW, srcH,
                             l0, 0.0f, t0, r0, 1.0f, t0, l1, 0.0f, t1);
            rasterizeQuadTri(dstBuffer.data(), bmpW, bmpH, stride, bgraOrder, srcRGBA, srcW, srcH,
                             r0, 1.0f, t0, r1, 1.0f, t1, l1, 0.0f, t1);
        } else {
            rasterizeQuadTri(dstBuffer.data(), bmpW, bmpH, stride, bgraOrder, srcRGBA, srcW, srcH,
                             l0, t0, 1.0f, r0, t0, 0.0f, l1, t1, 1.0f);
            rasterizeQuadTri(dstBuffer.data(), bmpW, bmpH, stride, bgraOrder, srcRGBA, srcW, srcH,
                             r0, t0, 0.0f, r1, t1, 0.0f, l1, t1, 1.0f);
        }
    }

    res.minX = minX; res.minY = minY; res.width = logW; res.height = logH;
    res.bmpW = bmpW; res.bmpH = bmpH;
    res.valid = true;
    return res;
}

} // namespace partwarp
