// MurmurHash3 x86_32, as used by scikit-learn's FeatureHasher / HashingVectorizer
// (sklearn.utils.murmurhash.murmurhash3_bytes_s32 with seed 0, signed result).
// Text is hashed as UTF-8 bytes; the column index is abs(hash) % n_features.

const encoder = new TextEncoder();

export function murmurhash3_32(bytes, seed = 0) {
  const len = bytes.length;
  const nblocks = len >>> 2;
  let h1 = seed | 0;
  const c1 = 0xcc9e2d51;
  const c2 = 0x1b873593;

  for (let i = 0; i < nblocks; i++) {
    const j = i * 4;
    let k1 = bytes[j] | (bytes[j + 1] << 8) | (bytes[j + 2] << 16) | (bytes[j + 3] << 24);
    k1 = Math.imul(k1, c1);
    k1 = (k1 << 15) | (k1 >>> 17);
    k1 = Math.imul(k1, c2);
    h1 ^= k1;
    h1 = (h1 << 13) | (h1 >>> 19);
    h1 = (Math.imul(h1, 5) + 0xe6546b64) | 0;
  }

  let k1 = 0;
  const tail = nblocks * 4;
  switch (len & 3) {
    case 3:
      k1 ^= bytes[tail + 2] << 16;
    // falls through
    case 2:
      k1 ^= bytes[tail + 1] << 8;
    // falls through
    case 1:
      k1 ^= bytes[tail];
      k1 = Math.imul(k1, c1);
      k1 = (k1 << 15) | (k1 >>> 17);
      k1 = Math.imul(k1, c2);
      h1 ^= k1;
  }

  h1 ^= len;
  h1 ^= h1 >>> 16;
  h1 = Math.imul(h1, 0x85ebca6b);
  h1 ^= h1 >>> 13;
  h1 = Math.imul(h1, 0xc2b2ae35);
  h1 ^= h1 >>> 16;
  return h1 | 0; // signed int32, like sklearn's *_s32 variant
}

/** Column index for a token, matching sklearn: abs(signed hash) % n_features. */
export function hashIndex(token, nFeatures) {
  const h = murmurhash3_32(encoder.encode(token), 0);
  return Math.abs(h) % nFeatures;
}
