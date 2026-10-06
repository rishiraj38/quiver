#include "textflag.h"

// The Go assembler has no mnemonics for the vector float ops used here, so
// they are emitted as raw words. Encodings (4 x float32 lanes):
//
//	fsub  Vd, Vn, Vm = 0x4EA0D400 | m<<16 | n<<5 | d
//	fadd  Vd, Vn, Vm = 0x4E20D400 | m<<16 | n<<5 | d
//	fmla  Vd, Vn, Vm = 0x4E20CC00 | m<<16 | n<<5 | d    (Vd += Vn * Vm)
//	faddp Vd, Vn, Vm = 0x6E20D400 | m<<16 | n<<5 | d
//
// Both kernels keep four accumulators (V0-V3) so the multiply-adds of one
// iteration do not wait on each other, and process 16 floats per iteration,
// then 4 at a time, then a scalar tail.

// func l2SquaredNEON(a, b *float32, n int) float32
TEXT ·l2SquaredNEON(SB), NOSPLIT, $0-28
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD n+16(FP), R2
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

l2_loop16:
	CMP  $16, R2
	BLT  l2_loop4
	VLD1.P 64(R0), [V4.S4, V5.S4, V6.S4, V7.S4]
	VLD1.P 64(R1), [V16.S4, V17.S4, V18.S4, V19.S4]
	WORD $0x4EB0D484 // fsub v4, v4, v16
	WORD $0x4EB1D4A5 // fsub v5, v5, v17
	WORD $0x4EB2D4C6 // fsub v6, v6, v18
	WORD $0x4EB3D4E7 // fsub v7, v7, v19
	WORD $0x4E24CC80 // fmla v0, v4, v4
	WORD $0x4E25CCA1 // fmla v1, v5, v5
	WORD $0x4E26CCC2 // fmla v2, v6, v6
	WORD $0x4E27CCE3 // fmla v3, v7, v7
	SUB  $16, R2
	B    l2_loop16

l2_loop4:
	CMP  $4, R2
	BLT  l2_reduce
	VLD1.P 16(R0), [V4.S4]
	VLD1.P 16(R1), [V16.S4]
	WORD $0x4EB0D484 // fsub v4, v4, v16
	WORD $0x4E24CC80 // fmla v0, v4, v4
	SUB  $4, R2
	B    l2_loop4

l2_reduce:
	WORD $0x4E21D400 // fadd v0, v0, v1
	WORD $0x4E23D442 // fadd v2, v2, v3
	WORD $0x4E22D400 // fadd v0, v0, v2
	WORD $0x6E20D400 // faddp v0, v0, v0
	WORD $0x6E20D400 // faddp v0, v0, v0

l2_tail:
	CBZ   R2, l2_done
	FMOVS (R0), F4
	FMOVS (R1), F5
	FSUBS F5, F4, F4
	FMULS F4, F4, F4
	FADDS F4, F0, F0
	ADD   $4, R0
	ADD   $4, R1
	SUB   $1, R2
	B     l2_tail

l2_done:
	FMOVS F0, ret+24(FP)
	RET

// func dotNEON(a, b *float32, n int) float32
TEXT ·dotNEON(SB), NOSPLIT, $0-28
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD n+16(FP), R2
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

dot_loop16:
	CMP  $16, R2
	BLT  dot_loop4
	VLD1.P 64(R0), [V4.S4, V5.S4, V6.S4, V7.S4]
	VLD1.P 64(R1), [V16.S4, V17.S4, V18.S4, V19.S4]
	WORD $0x4E30CC80 // fmla v0, v4, v16
	WORD $0x4E31CCA1 // fmla v1, v5, v17
	WORD $0x4E32CCC2 // fmla v2, v6, v18
	WORD $0x4E33CCE3 // fmla v3, v7, v19
	SUB  $16, R2
	B    dot_loop16

dot_loop4:
	CMP  $4, R2
	BLT  dot_reduce
	VLD1.P 16(R0), [V4.S4]
	VLD1.P 16(R1), [V16.S4]
	WORD $0x4E30CC80 // fmla v0, v4, v16
	SUB  $4, R2
	B    dot_loop4

dot_reduce:
	WORD $0x4E21D400 // fadd v0, v0, v1
	WORD $0x4E23D442 // fadd v2, v2, v3
	WORD $0x4E22D400 // fadd v0, v0, v2
	WORD $0x6E20D400 // faddp v0, v0, v0
	WORD $0x6E20D400 // faddp v0, v0, v0

dot_tail:
	CBZ   R2, dot_done
	FMOVS (R0), F4
	FMOVS (R1), F5
	FMULS F5, F4, F4
	FADDS F4, F0, F0
	ADD   $4, R0
	ADD   $4, R1
	SUB   $1, R2
	B     dot_tail

dot_done:
	FMOVS F0, ret+24(FP)
	RET
