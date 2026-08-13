// bits — 32-bit bitwise operations for Typst, built from integer arithmetic.
//
// Typst has no bitwise operators, but they all reduce to arithmetic on 32-bit words:
//   • shifts / rotates ARE base-2 arithmetic — multiply / divide by powers of two;
//     a rotate's two halves occupy disjoint bit positions, so OR becomes +.
//   • and / or / xor have no arithmetic shortcut, so they use a precomputed 4-bit
//     lookup table (8 table hits per 32-bit op instead of 32 single-bit tests).
//   • the i64 ceiling (2^63) overflows a full 32×32 product, so `mul` and `mulhi`
//     multiply in 16-bit chunks.
// Enough to implement counter RNGs, hashes and checksums (Threefry, PCG, Murmur, …).
//
//   #import "vendor/chalkdust/bits/lib.typ" as bits
//   #bits.bxor(0xF0F0, 0x00FF)      // 0xF00F
//   #bits.rotl(0x80000001, 1)       // 0x00000003

#let W = 4294967296              // 2^32
#let _word(x, name: "value") = {
  assert(type(x) == int, message: "bits." + name + ": expected an integer")
  calc.rem(calc.rem(x, W) + W, W)
}
#let mask(x) = _word(x)          // keep the low 32 bits, including for negative inputs

// ── shifts & rotates (free — powers of two) ──
#let _shift(k) = {
  assert(type(k) == int and k >= 0, message: "bits: shift count must be a non-negative integer")
  k
}
#let shl(x, k) = {
  let x = _word(x)
  let k = _shift(k)
  if k >= 32 { 0 } else { calc.rem(x * calc.pow(2, k), W) }
}
#let shr(x, k) = {
  let x = _word(x)
  let k = _shift(k)
  if k >= 32 { 0 } else { calc.floor(x / calc.pow(2, k)) }
}
#let rotl(x, k) = {
  let x = _word(x)
  assert(type(k) == int, message: "bits: rotate count must be an integer")
  let k = calc.rem(calc.rem(k, 32) + 32, 32)
  if k == 0 { x }
  else { calc.rem(x * calc.pow(2, k), W) + calc.floor(x / calc.pow(2, 32 - k)) }
}
#let rotr(x, k) = rotl(x, -k)
#let bnot(x) = W - 1 - _word(x)

// ── and / or / xor via 4-bit nibble tables ──
#let _nib(f) = range(16).map(a => range(16).map(b => {
  let r = 0
  let p = 1
  let x = a
  let y = b
  for _ in range(4) {
    if f(calc.rem(x, 2), calc.rem(y, 2)) == 1 { r += p }
    x = calc.floor(x / 2)
    y = calc.floor(y / 2)
    p = p * 2
  }
  r
}))
#let _XOR = _nib((a, b) => if a != b { 1 } else { 0 })
#let _AND = _nib((a, b) => a * b)
#let _OR = _nib((a, b) => if a + b > 0 { 1 } else { 0 })
#let _apply(tbl, a, b) = {
  let r = 0
  let p = 1
  let x = _word(a)
  let y = _word(b)
  for _ in range(8) {                                 // 8 nibbles = 32 bits
    r += tbl.at(calc.rem(x, 16)).at(calc.rem(y, 16)) * p
    x = calc.floor(x / 16)
    y = calc.floor(y / 16)
    p = p * 16
  }
  r
}
#let bxor(a, b) = _apply(_XOR, a, b)
#let band(a, b) = _apply(_AND, a, b)
#let bor(a, b) = _apply(_OR, a, b)

// ── 32-bit modular arithmetic (overflow-safe) ──
#let add(a, b) = mask(_word(a) + _word(b))
#let sub(a, b) = mask(_word(a) - _word(b))
#let mul(a, b) = {                                    // (a·b) mod 2^32
  let a = _word(a)
  let b = _word(b)
  let al = calc.rem(a, 65536)
  let ah = calc.floor(a / 65536)
  let bl = calc.rem(b, 65536)
  let bh = calc.floor(b / 65536)
  calc.rem(al * bl + calc.rem(al * bh + ah * bl, 65536) * 65536, W)
}
#let mulhi(a, b) = {                                  // high 32 bits of the 64-bit product
  let a = _word(a)
  let b = _word(b)
  let al = calc.rem(a, 65536)
  let ah = calc.floor(a / 65536)
  let bl = calc.rem(b, 65536)
  let bh = calc.floor(b / 65536)
  ah * bh + calc.floor((al * bl + (al * bh + ah * bl) * 65536) / W)
}
