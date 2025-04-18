/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */
#include "stream.h"

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

__device__ float6 computeCov2D_halfgs(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, float6 view_cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002).
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	// Change J for half gaussian
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		(focal_x / t.z), 0.0, (-(focal_x * t.x) / (t.z * t.z)),
		0.0, (focal_y / t.z), (-(focal_y * t.y) / (t.z * t.z)),
		0, 0, (focal_y / t.z));

	glm::mat3 W = glm::mat3(
		(viewmatrix[0]), (viewmatrix[4]), (viewmatrix[8]),
		(viewmatrix[1]), (viewmatrix[5]), (viewmatrix[9]),
		(viewmatrix[2]), (viewmatrix[6]), (viewmatrix[10]));

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

    // save the result without 2D projection, to use in the cutting plane calculation
    glm::mat3 rot_cov = glm::transpose(W) * glm::transpose(Vrk) * W;

    view_cov3D.x = rot_cov[0][0];
    view_cov3D.y = rot_cov[0][1];
    view_cov3D.z = rot_cov[1][1];
    view_cov3D.w = rot_cov[0][2];
    view_cov3D.u = rot_cov[1][2];
    view_cov3D.v = rot_cov[2][2];

//     glm::mat3 cov = glm::transpose(J) * glm::transpose(rot_cov) * J;

    // calculate the splatting 2D gaussian covariance
	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	cov[2][2] += 0.3f;
	return {cov[0][0], cov[0][1], cov[1][1], cov[0][2], cov[1][2], cov[2][2]};//{ float(cov[0][0]), float(cov[0][1]), float(cov[1][1]), float(cov[0][2]), float(cov[1][2]), float(cov[2][2])};
}

__device__ void generate_type(float3 cov, float3 cov_small, float op1, float op2, float& type, float3 normal, float x_term, float y_term, float3 conic)
{

    float full = (fabsf(op1-op2)<0.0039)?0.f:1.f; //if type is 0, then we use the whole gaussian
    //find the angle between vec and [0,1], which will tell us diagional
    //diagional = vec.x*vec.y<0.0f ? 1.0f : 0.0f; //use mean as center, vector in diagional(1,4) or in anti-diagional(2,3)
    float op_inverse = op1 > op2 ? 1.0f: 0.0f;

    float cx = conic.x;
    float cy = conic.y;
    float cz = conic.z;
    float det2 = 1/sqrtf(cx*cy*cy - 2*cx*cy*cy + cz*cx*cx);
    float2 point3 = {cy*det2, -cx*det2};
    float2 point1 = {-cy*det2, cx*det2};
    det2 = 1/sqrtf(cx*cz*cz - 2*cz*cy*cy + cz*cy*cy);
    float2 point2 = {cz*det2, -cy*det2};
    float2 point4 = {-cz*det2, cy*det2};

    float part1 = (normal.x*1.4142135f+x_term);
    float part2 = (normal.y*1.4142135f+y_term);
    float point1_large= (part1*point1.x+part2*point1.y) > 0.f?1.0f-op_inverse:0.f+op_inverse;
    float point2_large= (part1*point2.x+part2*point2.y) > 0.f?1.0f-op_inverse:0.f+op_inverse;
    float point3_large= 1.f - point1_large;//(part1*point3.x+part2*point3.y) > 0.f?1.0f-op_inverse:0.f+op_inverse;
    float point4_large= 1.f - point2_large;//(part1*point4.x+part2*point4.y) > 0.f?1.0f-op_inverse:0.f+op_inverse;

    type = full*(point1_large*point4_large*(1-point2_large)*(1-point3_large)*1.f + point1_large*point2_large*(1-point4_large)*(1-point3_large)*2.f + point3_large*point4_large*(1-point2_large)*(1-point1_large)*3.f+point2_large*point3_large*(1-point1_large)*(1-point4_large)*4.f);
//     type = full*(diagional*1.0f*leftright + (1.0f-diagional)*2.0f*(1.0f-leftright) + (1.0f-diagional)*3.0f*leftright + diagional*4.0f*(1.0f-leftright));
}



// this function is used to calculate the covariance for a 3D ellipsoid through the cutting plane, then we can splat this ellipsoid to 2D and generate the projection for the 3D cutting plane
__device__ void calculate_small(float6 cov_input, const float* viewmatrix, float3 normal, float* cov_new_small) {

    glm::vec3 v1, v2;

    glm::vec3 n(normal.x, normal.y, normal.z);
    n = glm::normalize(n);

    // initialize the matrix
    glm::mat3 cov;
    cov[0][0] = cov_input.x; // cov(0,0)
    cov[0][1] = cov[1][0] = cov_input.y; // cov(0,1) and cov(1,0)
    cov[0][2] = cov[2][0] = cov_input.w; // cov(0,2) and cov(2,0)
    cov[1][1] = cov_input.z; // cov(1,1)
    cov[1][2] = cov[2][1] = cov_input.u; // cov(1,2) and cov(2,1)
    cov[2][2] = cov_input.v; // cov(2,2)


    // initialize the basis
    if (n.x == 0.0f && n.y == 0.0f) {
        v1 = glm::vec3(1.0f, 0.0f, 0.0f);
        v2 = glm::vec3(0.0f, 1.0f, 0.0f);

    } else {
        // normalize n
        //
        v1 = glm::normalize(glm::vec3(n.y, -n.x, 0.0f));  //
        v2 = glm::normalize(glm::cross(n, v1));            //
    }

    // construct transformation matrix for basis
    glm::mat3 basis(v1, v2, n);
    glm::mat3 R_transform = glm::transpose(basis);  //


    // transform the basis
    glm::mat3 cov_transformed = R_transform * cov * basis;

    // extract elements
    float a = cov_transformed[0][0];
    float b = cov_transformed[0][1];
    float c = cov_transformed[1][1];
    float d = cov_transformed[0][2];
    float e = cov_transformed[1][2];
    float f = cov_transformed[2][2];

    float div = 1/f;

    // use Schur complement to calculate the covariance for cutting plane
    glm::mat3 cov_new_3d(
        a-d*(d*div), b-d*(e*div), 0.0f,
        b-d*(e*div), c-e*(e*div), 0.0f,
        0.0f, 0.0f, max(a-d*(d*div), c-e*(e*div)) / (100.0f));
    //0.0f, 0.0f, lambda / (100.0f));
    // back to the camera coordinate
    cov_new_3d = basis * cov_new_3d * R_transform;

    // save the result
    cov_new_small[0] = cov_new_3d[0][0];
    cov_new_small[1] = cov_new_3d[0][1];
    cov_new_small[2] = cov_new_3d[0][2];
    cov_new_small[3] = cov_new_3d[1][1];
    cov_new_small[4] = cov_new_3d[1][2];
    cov_new_small[5] = cov_new_3d[2][2];
}




// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	float* radii_comp,

	// hgs 相关
	const bool hgs,
	const float* hgs_normals,
	float3* save_normal,
	float* cov3D_smalls,
	float4* conic_opacity1,
	float4* conic_opacity2,
	uint4* conic_opacity3,
	uint4* conic_opacity4,
	float3* conic_opacity5,
	uint4* conic_opacity6)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;	// 每个高斯点的 3D 协方差矩阵 指针，3D 协方差矩阵用 6 个元素表示
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	if (hgs)
	{	
		// 转化 3D 法向量
		float3 gs_normal = {hgs_normals[3*idx],hgs_normals[3*idx+1],hgs_normals[3*idx+2]};
		gs_normal = transformPoint4x3(gs_normal, viewmatrix);

		float3 cov;
		float6 view_cov3D;
		float6 cov_temp = computeCov2D_halfgs(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, view_cov3D, viewmatrix);
		cov = {cov_temp.x, cov_temp.y, cov_temp.z};

		float det = cov.x * cov.z - cov.y * cov.y; //det = (cov.x * cov.z - cov.y * cov.y);
		if (det == 0.0f)
			return;
		//float det_inv = 1.f / sqrt(det);
		float det_inv2 = 1.f / det;

		float3 conic = { cov.z * det_inv2, -cov.y * det_inv2, cov.x * det_inv2 };//conic is the inverse of the variance matrix for 2D
		//////calculate rectangle size for 2D ellipse:
		float power = logf(256.f * max(opacities[2 * idx], opacities[2 * idx+1]));//logf(2.f) * 8.0f + logf(2.f) * log2_opacity;
		int width = (int)(1.414214f * __fsqrt_rn(cov.x * power) + 1.0f);
		int height = (int)(1.414214f * __fsqrt_rn(cov.z * power) + 1.0f);

		float3 cov_small;
		const float* cov3D_small;

		// 如果只覆盖一个 tile，则不用再计算小高斯
		if (2*width > BLOCK_X || 2*height > BLOCK_Y) //cov3D_precomp_small != nullptr &&  && (fabsf(opacities[2 * idx]-opacities[2 * idx+1])>0.004f)
		{
			// calculate_small(cov3D_inv, viewmatrix, gs_normal, cov3D_smalls + idx * 6);
			calculate_small(cov_temp, viewmatrix, gs_normal, cov3D_smalls + idx * 6);
			// calculate_small(view_cov3D, viewmatrix, gs_normal, cov3D_smalls + idx * 6, p_orig, focal_x, focal_y, tan_fovx, tan_fovy);
			cov3D_small = cov3D_smalls + idx * 6;
			// cov3D_small = cov3D_precomp_small + idx * 6;
			cov_small = {cov3D_small[0],cov3D_small[1],cov3D_small[3]};//computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D_small, viewmatrix);
			// cov_small = cov;
		}
		else{
			cov_small = cov;
		}
		// 计算 2D 协方差矩阵的特征值
		float lam1 = cov_small.x;//cov_small.x >= cov_small.z?small_lam1:small_lam2;
		float lam2 = cov_small.z;//cov_small.x < cov_small.z?small_lam1:small_lam2;

		// calculate the rectangle around each half for the half gaussian
		int width_small = (int)(1.414214f * __fsqrt_rn(lam1 * power) + 1.0f);
		int height_small = (int)(1.414214f * __fsqrt_rn(lam2 * power) + 1.f);

		width_small = min(width_small, width);
		height_small = 2*height_small>height ? min(height_small, height):max(height_small, height);//((float)height_small/(float)height) > 0.35f ?min(height_small, height):max(height_small, height); //min(height_small, height);//
		if (width <= 0 || height <= 0){
			return;
		}

		// height and width for the reactangle for the other half
		float power2 = logf(256.f * min(opacities[2 * idx], opacities[2 * idx+1]));//logf(2.f) * 8.0f + logf(2.f) * log2_opacity;
		int width_another = (int)(1.414214f * __fsqrt_rn(cov.x * power2) + 1.0f);
		int height_another = (int)(1.414214f * __fsqrt_rn(cov.z * power2) + 1.0f);

		int width_small_another = (int)(1.414214f * __fsqrt_rn(lam1 * power2) + 1.0f);
		int height_small_another = (int)(1.414214f * __fsqrt_rn(lam2 * power2) + 1.f);

		width_small_another = min(width_small_another, width_another);
		height_small_another = 2*height_small_another>height_another?min(height_small_another, height_another):max(height_small_another, height_another);//((float)height_small_another/(float)height_another) > 0.35f ?min(height_small_another, height_another):max(height_small_another, height_another); //min(height_small_another, height_another);//

		//////
		// Compute extent in screen space (by finding eigenvalues of
		// 2D covariance matrix). Use extent to compute a bounding rectangle
		// of screen-space tiles that this Gaussian overlaps with. Quit if
		// rectangle covers 0 tiles.

		float mid = 0.5f * (cov.x + cov.z);
		float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
		//float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
		float my_radius = ceil(3.f * sqrt(lambda1));//max(lambda1, lambda2)));
		// 	float my_radius = sqrtf(max(width_small, width)*max(width_small, width)+max(height_small, height)*max(height_small, height));

		float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
		uint2 rect_min, rect_max;
		uint2 rect_min_another, rect_max_another;
		float type;

		// 计算 I(x,y)的参数：
		// 计算分母，sqrt(2)·sigmaz|xy， sigmaz|xy <-> 当前 xy平面下, z的方差，
		float first_divide = 1/(1.4142135f*sqrtf(max(0.00000001f,cov_temp.v - (cov_temp.w*cov_temp.w* conic.x+2*cov_temp.w*cov_temp.u*conic.y+cov_temp.u*cov_temp.u* conic.z))));
		// 计算当前 xy平面下，z的均值 miuz|xy
		float x_term = cov_temp.w* conic.x + cov_temp.u* conic.y;
		float y_term = cov_temp.u* conic.z + cov_temp.w* conic.y;
		gs_normal.z = 1.f/(1.4142135f*(gs_normal.z+0.000001f)); //just in case z is 0
		gs_normal.x = gs_normal.x*gs_normal.z;
		gs_normal.y = gs_normal.y*gs_normal.z;
		generate_type(cov, cov_small, opacities[2 * idx], opacities[2 * idx+1], type, gs_normal, x_term, y_term, conic);
		
		hgs_getRect_final(point_image, width, height, width_small, height_small, width_another, height_another, width_small_another, height_small_another, (int)type, rect_min, rect_max, rect_min_another, rect_max_another, grid);
		if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
			return;

		// If colors have been precomputed, use them, otherwise convert
		// spherical harmonics coefficients to RGB color.
		if (colors_precomp == nullptr)
		{
			glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
			rgb[idx * C + 0] = result.x;
			rgb[idx * C + 1] = result.y;
			rgb[idx * C + 2] = result.z;
		}

		// Store some useful helper data for the next steps.
		depths[idx] = p_view.z;
		radii[idx] = my_radius;
		points_xy_image[idx] = point_image;
		conic_opacity1[idx] = { conic.x, conic.y, conic.z, opacities[2 * idx]};
		///for new half gaussian function
		gs_normal.x = gs_normal.x*1.4142135f*first_divide;
		gs_normal.y = gs_normal.y*1.4142135f*first_divide;
		gs_normal.z = gs_normal.z*1.4142135f*first_divide;
		save_normal[idx] = gs_normal;
		// 储存 miuz|xy / sqrt(2)·sigmaz|xy , 和 另一个半高斯的透明度
		conic_opacity2[idx] = {x_term*first_divide, y_term*first_divide, first_divide, opacities[2 * idx + 1]};

		uint2 rect_min2, rect_max2;
		rect_min2.x = min(rect_min.x,rect_min_another.x);
		rect_min2.y = min(rect_min.y,rect_min_another.y);
		rect_max2.x = max(rect_max.x,rect_max_another.x);
		rect_max2.y = max(rect_max.y,rect_max_another.y);

		// 储存当前半高斯在 xy平面下的矩形范围, 后续不用再计算
		conic_opacity3[idx] = {rect_min2.x, rect_min2.y, rect_max2.x, rect_max2.y};
		//conic_opacity4[idx] = {rect_min.x, rect_min.y, rect_max.x, rect_max.y};
		conic_opacity5[idx] = {cov_temp.w,cov_temp.u,cov_temp.v};
		//conic_opacity6[idx] = {rect_min_another.x, rect_min_another.y, rect_max_another.x, rect_max_another.y};
		tiles_touched[idx] = (rect_max2.x-rect_min2.x)*(rect_max2.y-rect_min2.y);
	}
	// 非 hgs 渲染
	else{
		float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);
		/*
		* 源代码中近期加入了 抗锯齿处理
		* 原理就是：增大高斯的协方差，将高斯范围扩大（模糊），然后对不透明度做一个修正：
		* opacity * h_convolution_scaling
		* h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov))
		* det_cov = cov.x * cov.z - cov.y * cov.y    原协方差行列式
		* h_var = 0.3f
		* cov.x += h_var;
		* cov.z += h_var;
		* det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;	模糊后协方差行列式
		*/


		// Invert covariance (EWA algorithm)
		float det = (cov.x * cov.z - cov.y * cov.y);
		if (det == 0.0f)
			return;
		float det_inv = 1.f / det;
		float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

		// Compute extent in screen space (by finding eigenvalues of
		// 2D covariance matrix). Use extent to compute a bounding rectangle
		// of screen-space tiles that this Gaussian overlaps with. Quit if
		// rectangle covers 0 tiles. 
		float mid = 0.5f * (cov.x + cov.z);
		float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
		float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
		float my_radius = 3.f * sqrt(max(lambda1, lambda2));
		float my_radius_ceil = ceil(my_radius);
		float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
		uint2 rect_min, rect_max;
		// 用 圆心 加减 3*最长轴，得到 矩形范围，其实相当于先获得圆，然后获得包裹圆的矩形
		getRect(point_image, my_radius_ceil, rect_min, rect_max, grid);
		if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
			return;

		// If colors have been precomputed, use them, otherwise convert
		// spherical harmonics coefficients to RGB color.
		if (colors_precomp == nullptr)
		{
			glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
			rgb[idx * C + 0] = result.x;
			rgb[idx * C + 1] = result.y;
			rgb[idx * C + 2] = result.z;
		}

		// Store some useful helper data for the next steps.
		// change the z dist to the dist to camera center
		// 这里计算的是 高斯点到相机中心的距离，原代码是 z 轴距离，因为我们要计算向光源点的遮挡，深度排序
		depths[idx] = p_view.z*p_view.z+p_view.y*p_view.y+p_view.x*p_view.x;
		radii[idx] = my_radius_ceil;
		radii_comp[idx] = my_radius;
		points_xy_image[idx] = point_image;
		// Inverse 2D covariance and opacity neatly pack into one float4
		// conic.x, conic.y, conic.z 2D 协方差矩阵的逆，conic.w 高斯点的不透明度
		conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
		tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	}
}


// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
/**
 * @brief forward 渲染
 * 
 * @attention __global__ ———— 在 GPU 上执行的函数
 * @attention __launch_bounds__ ———— 告诉编译器，限制每个 block 的线程数量，保证每个线程有更多的硬件资源可以调用，比如寄存器
 * @attention __restrict__ ———— 编译器默认：指针可能重叠
 * 								- 告诉编译器，这个指针指向的数据不会和其他 __restrict__ 指针指向的数据重叠，方便编译器内存优化
 * @attention const ———— 告诉编译器变量只读，方便编译器内存优化
 * @attention tile <-> block，pixel <-> thread
 * 
 * @else CHANNELS ———— 模板参数：颜色通道数
 */
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
original_renderCUDA(
	const uint2* __restrict__ ranges,	// 每个 tile 的 point_list 索引范围
	const uint32_t* __restrict__ point_list,	// 按[tile | depth] 键值对 排序后的 渲染高斯点 索引列表
	int W, int H,	// 图片宽度和高度
	const float2* __restrict__ points_xy_image,	// 每个高斯点的 二维坐标
	const float* __restrict__ depths,	// 每个高斯点的 深度
	const float* __restrict__ features,	// 每个高斯点的 sh特征
	const float4* __restrict__ conic_opacity,	// 每个高斯点的 椭圆参数
	float* __restrict__ final_T,	// 每个像素的 accum_alpha
	uint32_t* __restrict__ n_contrib,	// 每个像素的 贡献者数量
	const float* __restrict__ bg_color,	// 背景颜色
	
	// 输出
	float* __restrict__ out_color,	// 每个像素的 颜色
	float* __restrict__ out_weight,	// 每个像素的 权重
	float* __restrict__ out_trans,	// 每个像素的 透明度

	// 其他参数
	float* __restrict__ non_trans,
	const float offset,	// 0.015
	const float thres,	// 4
	const bool is_train,
	
	// added
	const float* __restrict__ radii_comp
	)
{
	// Identify current tile and associated min/max pixel range.
	// 启动所有像素的线程，每个线程处理一个像素

	// 获取线程在当前块的局部索引，通过 block.thread_index().x 和 block.thread_index().y 获取
	// block.group_index().x 和 block.group_index().y 获取当前块的	二维全局索引
	auto block = cg::this_thread_block();
	// 计算 tile 的行数，向上取整
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	// pix_min = {当前块的索引x * 块width，当前块的索引y * 块height} = 该块的左上角（第一个像素坐标）
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	// 因为块是向上取整，因此最大像素可能在图片外，所以需要取图片的宽度和高度作为最小值
	// pix_max = {min(当前块的索引x * 块width + 块width，图像宽度)，min(当前块的索引y * 块height + 块height，图像高度)} = 考虑图片下的该块的右下角（最后一个像素坐标）
	// ~~~~ pixmax 暂时后续其实没有用到，所以 pix 还是需要判断是否在图片内
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	// 根据像素的局部索引，计算当前像素的 二维全局坐标
	// 当前 pix = {pix_min.x + 局部坐标x，pix_min.y + 局部坐标.y}
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	// 计算当前像素的线性索引 idx
	uint32_t pix_id = W * pix.y + pix.x;
	// 用 float 保存当前像素 二维坐标
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	// 判断当前像素是否在图片内
	bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	// 如果不在，则不进行渲染，done = true
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	/* 回忆 ranges 的含义：
	 * 每个元素表示一个 tile 的按 深度排序后的 渲染高斯点 顺序索引范围
	 */ 
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	// 当前块需要处理的 循环次数，每一次循环考虑 BLOCK_SIZE 个 渲染高斯点，向上取整
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	// 当前块总共需要处理的 渲染高斯点 数量
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	//！！！核心 ———— 共享内存 ！！！
	/**
	 * @attention __shared__ ———— 共享内存变量声明，表示这些变量是在一个线程块（block）内所有线程共享的一段内存
	 * @attention BLOCK_SIZE ———— 每个线程块的线程数量
	 */

	// original
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	
	// bdrf
	__shared__ float collected_depth[BLOCK_SIZE];
	__shared__ int collected_id_pointT[BLOCK_SIZE];
	__shared__ float collected_depth_pointT[BLOCK_SIZE];
	__shared__ float2 collected_xy_pointT[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity_pointT[BLOCK_SIZE];

	// smoother
	__shared__ float collected_radii_comp[BLOCK_SIZE];


	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	uint32_t contributor_pointT = 0;
	uint32_t last_contributor_pointT = 0;
	float w = T;
	// float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	// offset point id
	int pointT=0;
	// offset point depth
	float depthT=0.0;
	int i_pointT=0;
	int progress_pointT = i_pointT * BLOCK_SIZE + block.thread_rank();

	// addded
	float temp = 1.0f;
	float depth_diff = 0.0f;
	float test_T2 = 0.0f;
	float alpha=0.0f;

	/**
	 * @brief 初始化 渲染高斯点 偏移索引
	 * @details 这里在干什么？
	 * 			前面初始化了公共变量，这里要把当前 tile 的 渲染高斯点加载到 共享内存中，供所有线程使用，
	 * 			所以进行了以下两点优化：
	 * 	 	    1. 当前块也就是tile的每个线程都并行加载一个 渲染高斯点 到 共享内存中，
	 *          2. 每次只考虑 线程数量 个 渲染高斯点，而不是 当前块的 渲染高斯点数量，因此有 range.x + progress_pointT < range.y
	 *          	- 这是完全合理的，因为我们的 渲染高斯点 是按深度排序并计算得，并且还有一些线程并不用考虑所有 渲染高斯点 就已经达到阈值，
	 * 				- 因此 渲染高斯点 按深度分批加载并去计算每个线程也就是像素的最终值。
	 * 
	 * @attention block.thread_rank() ———— 当前块中当前线程的线性索引
	 * @attention i_pointT ———— 当前块中所有线程共享的 渲染高斯点 批次偏移索引
	 * @attention progress_pointT ———— 当前块中所有线程共享的 渲染高斯点 偏移索引
	 */
	// 初始化 偏移点 相关数据，和 渲染高斯点无关
	// 下面会详细解释什么是偏移点
	if (range.x + progress_pointT < range.y)
	{
		int coll_id = point_list[range.x + progress_pointT]; // 取出 高斯属性索引
		collected_id_pointT[block.thread_rank()] = coll_id; // 写入 加载的 高斯点 属性索引
		collected_depth_pointT[block.thread_rank()] = depths[coll_id]; // 写入 高斯对应的深度信息
		collected_xy_pointT[block.thread_rank()] = points_xy_image[coll_id]; // 写入 高斯对应的 2D 坐标
		// conic_opacity[idx] = (conic.x, conic.y, conic.z 2D 协方差矩阵的逆，conic.w 高斯点的不透明度，不透明度)
		collected_conic_opacity_pointT[block.thread_rank()] = conic_opacity[coll_id]; // 写入 高斯对应的 2D 椭圆参数 
	}
	// 同步块内所有线程，确保所有线程都加载完 当前轮高斯点，等价于 __syncthreads()
	// block.sync() 
	block.sync();

	/**
	 * @brief 利用当前 渲染高斯点 批次，对各像素 alpha-blending
	 * 
	 * @param i ———— 当前块的 第 i 次循环
	 * @param toDo ———— 当前块总共 还需要处理的 渲染高斯点 数量
	 * @param rounds ———— 当前块需要处理的 循环次数，每一次循环考虑 BLOCK_SIZE 个 渲染高斯点，向上取整
	 * @param done ———— 当前线程/像素是否完成 当前批次任务
	 */
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		/**
		 * @brief 统计有多少个线程 done == true
		 * 
		 * @attention __syncthreads_count ———— 集体同步指令 + 条件统计
		 * @else __syncthreads_and ———— 集体同步指令 + 条件与，__syncthreads_or ———— 集体同步指令 + 条件或
		 */
		int num_done = __syncthreads_count(done);
		// 如果所有线程/像素都 done == true，即当前块完成渲染任务，则退出循环
		if (num_done == BLOCK_SIZE)
			break;


		// Collectively fetch per-Gaussian data from global to shared
		// 获取当前线程储存的 渲染高斯点 顺序索引
		int progress = i * BLOCK_SIZE + block.thread_rank();

		block.sync();
		
		// 加载当前批次 渲染高斯点，则将 该渲染高斯点 属性索引、深度、二维坐标、椭圆参数 写入 共享内存
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_depth[block.thread_rank()] = depths[coll_id];
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];	
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_radii_comp[block.thread_rank()] = thres ==-1 ? radii_comp[coll_id]*1.1f : thres; 
			
			// 写入 高斯对应的 用于对比的 椭圆半径

			
		}
		block.sync();

		// Else, iterate over current batch
		// 开始作战！！！
		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++)
		{	
			contributor++;
			
			// 当前线程/像素位置下，当前循环渲染高斯点密度 ———— power_j
			float2 xy_j = collected_xy[j];
			float2 d_j = { xy_j.x - pixf.x, xy_j.y - pixf.y };
			float dd=d_j.x*d_j.x+d_j.y*d_j.y;
			float4 con_o_j = collected_conic_opacity[j];
			float power_j = -0.5f * (con_o_j.x * d_j.x * d_j.x + con_o_j.z * d_j.y * d_j.y) - con_o_j.y * d_j.x * d_j.y;
			
			//vis_weight = exp(-(depth_j - depthT)² / σ²)
			//float vis_weight = exp(-(collected_depth[j] - depthT) * (collected_depth[j] - depthT) / (offset * offset));
			/**
			 * @brief 并没有采用逐个高斯点计算，而是采用了 基于 depth difference 的间歇式计算 
			 * 
			 * @details 可以认为高斯点只收到 offset 距离外的高斯点遮挡，而在 offset 区间内的高斯点对其影响忽略，即并不实时参与更新当前的遮挡值/透明度
			 * @attention 下面分支中判断部分会详细解释
			 * 
			 * @note 这是可以合理的，因为 3D高斯点 往往是互相覆盖的，
			 * 		 - 而采用实时逐个高斯去更新，不仅使计算量增大，并且更多不同像素之间  alpha blending 不产生更多的突变
			 * 		 - 控制「遮挡的敏感范围」（其实这里可以改进，高斯点附近的密度动态决定）
			 * 		 - 保证遮挡关系更柔和；* 
			 */
			// if now depth - offset point depth>offset
			depth_diff = collected_depth[j]-depthT;
			if(depth_diff>offset)	// depthT 刚开始为 0，所以第一个高斯点一定会进入循环
			{	
				while(1)
				{	
					
					T = T * temp;
					// 计算当前循环中 偏移点 的 密度
					float2 xy = collected_xy_pointT[pointT];
					float2 d = { xy.x - pixf.x, xy.y - pixf.y };
					float4 con_o = collected_conic_opacity_pointT[pointT];
					float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;

					// update offset point and offset point depth
					// 更新偏移点深度，刚开始更新为第一个高斯点深度
					depthT=collected_depth_pointT[pointT];
					pointT=pointT+1;

					/**
					 * @brief 如果偏移点索引超过 BLOCK_SIZE，说明可以开始下一批高斯的计算
					 * 
					 * @attention 这里的共享内存放的是 偏移点 的 高斯点属性索引、深度、二维坐标、椭圆参数，和当前高斯点无关
					 * @note 1. 偏移点 肯定 是在 当前高斯点 之前，所以不用担心 i_pointT，让他自己加加加
					 * 		 2. 并且所有线程统一从同样的高斯点开始，每个块共享一致的高斯点（不考虑范围，只考虑深度），因此偏移点的移动也是统一的，因此偏移点进入下一批的时机也是统一的
					 * 		 >>> 因此 不用担心死锁，也就是有个线程已经完成当前批次，却有的线程在这里等待
					 */ 
					if(pointT>=BLOCK_SIZE)
					{
						i_pointT+=1;
						if(i_pointT<rounds)	// 未超出当前块的渲染高斯点数量
						{
							block.sync();	// 等待所有线程 进入 更换偏移点批次状态，即偏移点需要向更深处移动
							// 获得当前线程需要加载的下一批中对应的 高斯点顺序索引
							progress_pointT = i_pointT * BLOCK_SIZE + block.thread_rank();
							// 由于向上取整，可能超出当前块的渲染高斯点数量，所以需要判断
							if (range.x + progress_pointT < range.y)
							{	
								// 获得当前线程需要加载的下一批中对应的 高斯点属性索引
								// 加载 高斯点属性索引、深度、二维坐标、椭圆参数
								int coll_id = point_list[range.x + progress_pointT];
								collected_id_pointT[block.thread_rank()] = coll_id;
								collected_depth_pointT[block.thread_rank()] = depths[coll_id];
								collected_xy_pointT[block.thread_rank()] = points_xy_image[coll_id];
								collected_conic_opacity_pointT[block.thread_rank()] = conic_opacity[coll_id];
							}
							block.sync();	// 等待所有线程 加载新的数据
						}
						pointT=0; // 下一批，因此回到起点，从0开始
					}
					contributor_pointT=contributor_pointT+1;

					// 计算 当前循环高斯点 的 密度/不透明度
					alpha=0.0;
					if (power <= 0.0f)	// 消除精度误差带来的影响
					{
						alpha = min(0.99f, con_o.w * exp(power));	// 增强过渡
						if (alpha < 1.0f / 255.0f)	// 过小剔除
							alpha=0.0;
					}
					w = alpha * T;	// 当前高斯点 的 贡献值
					// alpha ———— 密度 = 不透明度
					// 1-alpha ———— 当前循环点的 透明度/剩余透过率
					temp = (1 - alpha); // 准备进行一个 线性插值 来更新 T，因此 当前高斯点和偏移点 的 距离可能过近，因此进行适当平衡
					
					// T = T * temp ———— 考虑当前偏移点位置下的 遮挡 后，剩余透过率

					// 这里就是 间歇式判断 核心之一：
					/**
					 * @details：
					 * 进入当前循环的 高斯点在刚开始 一定是 和 偏移点 深度差值 大于 offset 的
					 * 因此肯定存在几轮循环中，不满足 collected_depth[j]-depthT<=offset 条件
					 * 所以此时，T = test_T，回到开头，偏移点向深处移动，直到满足当前高斯点 和 偏移点 深度差值 小于 offset 的条件
					 * （最深也就移动到当前高斯点，此时就像第一个高斯点一样，collected_depth[j]-depthT == 0）
					 * 此时，进行 alpha-blending 更新当前高斯点，也就是当前高斯点，考虑了之前所有高斯点的剩余透射率
					 * 此时，T 更新为 test_T，退出循环
					 * 
					 * ~~~ （j+1 代表相较于当前循环的 下一个高斯点）
					 * 
					 * 在第二轮大循环中，也就是下一个当前高斯点，他可能满足 collected_depth[j+1]-depthT>offset
					 * 那么此时 T 不更新，该高斯点继续只考虑上一个阶段的累积光照值，直到不满足 collected_depth[j+1]-depthT>offset
					 * 也就是说，此时 偏移点 需要进一步向深处移动，剩余透射率 T 也需要进行更新，
					 * 直到 再满足 collected_depth[j+1]-depthT<=offset
					 * 所以也就是 偏移点 在跟进 当前高斯点，两者之间时刻保持 offset 左右距离
					 * 
					 * ~~~
					 * 
					 * 所以说，这里采用了 基于 depth difference 的计算 
					 * 遮挡值/透明度 并不是实时更新，而是根据距离选择更新
					 * 		- 在总结一下，就是 偏移点 和 当前高斯点 的深度差值 小于 offset 时，一直采用上一个偏移点的剩余透射率，
					 * 		- 在 偏移点 和 当前高斯点 的深度差值 大于 offset 时，往深处移动偏移点，并更新 剩余透射率 T，
					 * 		- 直到 偏移点 和 当前高斯点 的深度差值 小于 offset 时，使用新偏移点的剩余透射率 T 进行计算
					 * >>> 也就是说，一直考虑 当前高斯点 offset 左右距离外 的剩余透射率，而不是考虑所有在自己前面的所有高斯点的遮挡
					*/ 

					// if offset point is close enough to current point
					depth_diff = collected_depth[j]-depthT;
					if(depth_diff<=offset)
					{
						if(dd<collected_radii_comp[j] && inside)
						{   
							atomicAdd(&(out_trans[collected_id[j]]), exp(power_j)*(T*(1-(depth_diff/offset) + (depth_diff/offset)*temp)));
							atomicAdd(&(non_trans[collected_id[j]]), exp(power_j));
						}
						break;
					}


					// T = test_T  // 原 code 位置
					// if(collected_depth[j]-depthT<=offset) {break;}
					


				}
			}

			// if offset point and current point is close enough
			// 这里就是 当高斯点与偏移点深度差值 小于 offset 时，则一直使用上一个阶段的累积透明度，进行光照值计算
			else{
				if(dd<collected_radii_comp[j] && inside)
				{
					atomicAdd(&(out_trans[collected_id[j]]), exp(power_j)*(T*(1-(depth_diff/offset) + (depth_diff/offset)*temp)));
					atomicAdd(&(non_trans[collected_id[j]]), exp(power_j));
				}
			}

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;

			// Record accumulated weights
			if(is_train && inside){
				atomicAdd(&out_weight[collected_id[j]], w);
			}
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
	}
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
hgs_renderCUDA(
	const uint2* __restrict__ ranges,	// 每个 tile 的 point_list 索引范围
	const uint32_t* __restrict__ point_list,	// 按[tile | depth] 键值对 排序后的 渲染高斯点 索引列表
	int W, int H,	// 图片宽度和高度
	const float2* __restrict__ points_xy_image,	// 每个高斯点的 二维坐标
	const float* __restrict__ depths,	// 每个高斯点的 深度
	const float* __restrict__ features,	// 每个高斯点的 sh特征
	const float4* __restrict__ conic_opacity,	// 每个高斯点的 椭圆参数
	float* __restrict__ final_T,	// 每个像素的 accum_alpha
	uint32_t* __restrict__ n_contrib,	// 每个像素的 贡献者数量
	const float* __restrict__ bg_color,	// 背景颜色
	
	// 输出
	float* __restrict__ out_color,	// 每个像素的 颜色
	float* __restrict__ out_weight,	// 每个像素的 权重
	float* __restrict__ out_trans,	// 每个像素的 透明度

	// 其他参数
	float* __restrict__ non_trans,
	const float offset,	// 0.015
	const float thres,	// 4
	const bool is_train,
	
	// added
	const float* __restrict__ radii_comp,
	
	// hgs 相关
	const bool hgs,
	const float* __restrict__ hgs_normals,
	const float* __restrict__ hgs_opacities,
	float* __restrict__ hgs_opacities_shadow,
	float* __restrict__ hgs_opacities_light,
	const float3* __restrict__ normal,
	const float4* __restrict__ conic_opacity1,
	const float4* __restrict__ conic_opacity2)
{

	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };


	bool inside = pix.x < W && pix.y < H;
	bool done = !inside;

	// 获取当前块的 渲染高斯点 顺序索引范围
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	// 当前块需要处理的 循环次数，每一次循环考虑 BLOCK_SIZE 个 渲染高斯点，向上取整
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	// 当前块总共需要处理的 渲染高斯点 数量
	int toDo = range.y - range.x;

	// original
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	
	// bdrf
	__shared__ float collected_depth[BLOCK_SIZE];
	__shared__ int collected_id_pointT[BLOCK_SIZE];
	__shared__ float collected_depth_pointT[BLOCK_SIZE];
	__shared__ float2 collected_xy_pointT[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity_pointT[BLOCK_SIZE];

	// smoother
	__shared__ float collected_radii_comp[BLOCK_SIZE];

	// hgs
	__shared__ float4 collected_conic_opacity1[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity2[BLOCK_SIZE];
	__shared__ float3 collected_normal[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	uint32_t contributor_pointT = 0;
	uint32_t last_contributor_pointT = 0;
	float w = T;
	// float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	// offset point id
	int pointT=0;
	// offset point depth
	float depthT=0.0;
	int i_pointT=0;
	int progress_pointT = i_pointT * BLOCK_SIZE + block.thread_rank();

	// addded
	float temp = 1.0f;
	float depth_diff = 0.0f;
	float test_T2 = 0.0f;
	float alpha=0.0f;


	if (range.x + progress_pointT < range.y)
	{
		int coll_id = point_list[range.x + progress_pointT]; // 取出 高斯属性索引
		collected_id_pointT[block.thread_rank()] = coll_id; // 写入 加载的 高斯点 属性索引
		collected_depth_pointT[block.thread_rank()] = depths[coll_id]; // 写入 高斯对应的深度信息
		collected_xy_pointT[block.thread_rank()] = points_xy_image[coll_id]; // 写入 高斯对应的 2D 坐标
		// conic_opacity[idx] = (conic.x, conic.y, conic.z 2D 协方差矩阵的逆，conic.w 高斯点的不透明度，不透明度)
		collected_conic_opacity_pointT[block.thread_rank()] = conic_opacity[coll_id]; // 写入 高斯对应的 2D 椭圆参数 
	}
	block.sync();

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{

		int num_done = __syncthreads_count(done);
		// 如果所有线程/像素都 done == true，即当前块完成渲染任务，则退出循环
		if (num_done == BLOCK_SIZE)
			break;


		// Collectively fetch per-Gaussian data from global to shared
		// 获取当前线程储存的 渲染高斯点 顺序索引
		int progress = i * BLOCK_SIZE + block.thread_rank();

		block.sync();
		
		// 加载当前批次 渲染高斯点，则将 该渲染高斯点 属性索引、深度、二维坐标、椭圆参数 写入 共享内存
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_depth[block.thread_rank()] = depths[coll_id];
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];	
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_radii_comp[block.thread_rank()] = thres ==-1 ? radii_comp[coll_id]*1.1f : thres; 
			
			// 写入 高斯对应的 用于对比的 椭圆半径

			
		}
		block.sync();

		// Else, iterate over current batch
		// 开始作战！！！
		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++)
		{	
			contributor++;
			
			// 当前线程/像素位置下，当前循环渲染高斯点密度 ———— power_j
			float2 xy_j = collected_xy[j];
			float2 d_j = { xy_j.x - pixf.x, xy_j.y - pixf.y };
			float dd=d_j.x*d_j.x+d_j.y*d_j.y;
			float4 con_o_j = collected_conic_opacity[j];
			float power_j = -0.5f * (con_o_j.x * d_j.x * d_j.x + con_o_j.z * d_j.y * d_j.y) - con_o_j.y * d_j.x * d_j.y;
			
			//vis_weight = exp(-(depth_j - depthT)² / σ²)
			//float vis_weight = exp(-(collected_depth[j] - depthT) * (collected_depth[j] - depthT) / (offset * offset));

			// if now depth - offset point depth>offset
			depth_diff = collected_depth[j]-depthT;
			if(depth_diff>offset)	// depthT 刚开始为 0，所以第一个高斯点一定会进入循环
			{	
				while(1)
				{	
					
					T = T * temp;
					// 计算当前循环中 偏移点 的 密度
					float2 xy = collected_xy_pointT[pointT];
					float2 d = { xy.x - pixf.x, xy.y - pixf.y };
					float4 con_o = collected_conic_opacity_pointT[pointT];
					float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;

					// update offset point and offset point depth
					// 更新偏移点深度，刚开始更新为第一个高斯点深度
					depthT=collected_depth_pointT[pointT];
					pointT=pointT+1;

					if(pointT>=BLOCK_SIZE)
					{
						i_pointT+=1;
						if(i_pointT<rounds)	// 未超出当前块的渲染高斯点数量
						{
							block.sync();	// 等待所有线程 进入 更换偏移点批次状态，即偏移点需要向更深处移动
							// 获得当前线程需要加载的下一批中对应的 高斯点顺序索引
							progress_pointT = i_pointT * BLOCK_SIZE + block.thread_rank();
							// 由于向上取整，可能超出当前块的渲染高斯点数量，所以需要判断
							if (range.x + progress_pointT < range.y)
							{	
								// 获得当前线程需要加载的下一批中对应的 高斯点属性索引
								// 加载 高斯点属性索引、深度、二维坐标、椭圆参数
								int coll_id = point_list[range.x + progress_pointT];
								collected_id_pointT[block.thread_rank()] = coll_id;
								collected_depth_pointT[block.thread_rank()] = depths[coll_id];
								collected_xy_pointT[block.thread_rank()] = points_xy_image[coll_id];
								collected_conic_opacity_pointT[block.thread_rank()] = conic_opacity[coll_id];
							}
							block.sync();	// 等待所有线程 加载新的数据
						}
						pointT=0; // 下一批，因此回到起点，从0开始
					}
					contributor_pointT=contributor_pointT+1;

					// 计算 当前循环高斯点 的 密度/不透明度
					alpha=0.0;
					if (power <= 0.0f)	// 消除精度误差带来的影响
					{
						alpha = min(0.99f, con_o.w * exp(power));	// 增强过渡
						if (alpha < 1.0f / 255.0f)	// 过小剔除
							alpha=0.0;
					}
					w = alpha * T;	// 当前高斯点 的 贡献值
					// alpha ———— 密度 = 不透明度
					// 1-alpha ———— 当前循环点的 透明度/剩余透过率
					temp = (1 - alpha); // 准备进行一个 线性插值 来更新 T，因此 当前高斯点和偏移点 的 距离可能过近，因此进行适当平衡
					
					// T = T * temp ———— 考虑当前偏移点位置下的 遮挡 后，剩余透过率

					// if offset point is close enough to current point
					depth_diff = collected_depth[j]-depthT;
					if(depth_diff<=offset)
					{
						if(dd<collected_radii_comp[j] && inside)
						{   
							atomicAdd(&(out_trans[collected_id[j]]), exp(power_j)*(T*(1-(depth_diff/offset) + (depth_diff/offset)*temp)));
							atomicAdd(&(non_trans[collected_id[j]]), exp(power_j));
						}
						break;
					}


					// T = test_T  // 原 code 位置
					// if(collected_depth[j]-depthT<=offset) {break;}		

				}
			}

			// if offset point and current point is close enough
			// 这里就是 当高斯点与偏移点深度差值 小于 offset 时，则一直使用上一个阶段的累积透明度，进行光照值计算
			else{
				if(dd<collected_radii_comp[j] && inside)
				{
					atomicAdd(&(out_trans[collected_id[j]]), exp(power_j)*(T*(1-(depth_diff/offset) + (depth_diff/offset)*temp)));
					atomicAdd(&(non_trans[collected_id[j]]), exp(power_j));
				}
			}

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;

			// Record accumulated weights
			if(is_train && inside){
				atomicAdd(&out_weight[collected_id[j]], w);
			}
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
	}
}


// 作用1：格式化
// 作用2：可以转换数据格式，满足函数签名，比如 float 为 const float*，核函数不支持 T 传入 const T

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* depths,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_weight,
	float* out_trans,
	float* non_trans,
	const float offset,
	const float thres,
	const bool is_train,
	const float* radii_comp,

	// hgs 相关
	const bool hgs,
	const float* hgs_normals,
	const float* hgs_opacities,
	float* hgs_opacities_shadow,
	float* hgs_opacities_light,
	const float3* normal,
	const float4* conic_opacity1,
	const float4* conic_opacity2)
{	
	// 瓦片并行渲染 -> 像素并行渲染
	// 启动 grid*block 个线程，也就是像素数量（可能大于图像像素数量，因为向上取整）
	// 每个 block 是一个 tile，每个线程处理 tile 中的一个像素，block 内线程共享内存协作完成该 tile 的像素渲染。
	if (hgs) 
	{
		hgs_renderCUDA<NUM_CHANNELS> << <grid, block, 0, MY_STREAM>> > ( 
			ranges,
			point_list,
			W, H,
			means2D,
			depths,
			colors,
			conic_opacity,
			final_T,
			n_contrib,
			bg_color,
			out_color,
			out_weight,
			out_trans,
			non_trans,
			offset,
			thres,
			is_train,
			radii_comp,

			// hgs 相关	
			hgs,
			hgs_normals,
			hgs_opacities,
			hgs_opacities_shadow,
			hgs_opacities_light,
			normal,
			conic_opacity1,
			conic_opacity2);

	}
	else
	{
		original_renderCUDA<NUM_CHANNELS> << <grid, block, 0, MY_STREAM>> > ( 
			ranges,
			point_list,
			W, H,
			means2D,
			depths,
			colors,
			conic_opacity,
			final_T,
			n_contrib,
			bg_color,
			out_color,
			out_weight,
			out_trans,
			non_trans,
			offset,
			thres,
			is_train,
			radii_comp);
	}
}
void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	float* radii_comp,
	
	// hgs 相关
	const bool hgs,
	const float* hgs_normals,
	float3* save_normal,
	float* cov3D_smalls,
	float4* conic_opacity1,
	float4* conic_opacity2,
	uint4* conic_opacity3,
	uint4* conic_opacity4,
	float3* conic_opacity5,
	uint4* conic_opacity6
	)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256, 0, MY_STREAM>> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		radii_comp,
		
		// hgs 相关
		hgs,
		hgs_normals,
		save_normal,
		cov3D_smalls,
		conic_opacity1,
		conic_opacity2,
		conic_opacity3,
		conic_opacity4,
		conic_opacity5,
		conic_opacity6
		);
}