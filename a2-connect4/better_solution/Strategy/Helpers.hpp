#ifndef FASTMATH_HPP
#define FASTMATH_HPP

#include <cstdint>

static inline float fastSqrt(float x) {
	union { float f; uint32_t i; } vx = { x };
	float xhalf = 0.5f*x;
	vx.i = 0x5f375a86 - (vx.i >> 1); // gives initial guess y0
	vx.f = vx.f * (1.5f - xhalf * vx.f*vx.f); // Newton step, repeating increases accuracy
	vx.f = vx.f * (1.5f - xhalf * vx.f*vx.f);
	vx.f = vx.f * (1.5f - xhalf * vx.f*vx.f);

	return 1 / vx.f;
}

static inline float fastLog(float x) {
	union { float f; uint32_t i; } vx = { x };
	float y = vx.i;
	return y * 8.2629582881927490e-8f - 87.989971088f;
}

#endif