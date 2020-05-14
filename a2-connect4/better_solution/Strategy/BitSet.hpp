#ifndef BITSET_HPP
#define BITSET_HPP
#include <cstdint>

typedef union {
	struct {
		uint64_t a0;
		uint64_t a1;
		uint64_t a2;
	} raw_data;
	uint64_t a[3];
} bits;

struct BitSet {
	static inline bool notZero(const bits& a) {
		return (a.a[0] | a.a[1] | a.a[2]) != 0;
	}

	static inline bits orWith(const bits& a, const bits& b) {
		return bits{ a.a[0] | b.a[0], a.a[1] | b.a[1], a.a[2] | b.a[2] };
	}

	static inline bits getBits(uint64_t a0, uint64_t a64, uint64_t a128) {
		return bits{ a0, a64, a128 };
	}

	static inline bits andWith(const bits& a, const bits& b) {
		return bits{ a.a[0] & b.a[0], b.a[1] & a.a[1], a.a[2] & b.a[2] };
	}

	static inline void set(bits &data, size_t pos) {
		data.a[pos >> 6] |= 1ULL << (pos & 63);
	}

	static inline bits rightShift(const bits& data, int x) {
		return bits{data.a[0] >> x | data.a[1] << (64 - x),
			data.a[1] >> x | data.a[2] << (64 - x), data.a[2] >> x};
	}

	static inline bool test(const bits& data, size_t pos) {
		return 1ULL << (pos & 63) & data.a[pos >> 6];
	}
};

#endif