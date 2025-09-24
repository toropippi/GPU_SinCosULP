// 1ディスパッチ = 2^24 個のワークアイテム想定（global_size = 1<<24）
// chunk_hi は 0..255。ビットパターン u = (chunk_hi << 24) | gid に対応。
// ビルドオプションは付けない（-cl-fast-relaxed-math 等なし）。

#define CHUNK_LOG2  24u
#define CHUNK_SIZE  (1u << CHUNK_LOG2)

__kernel void sweep_trig_4outs(
    __global float* out_sin,         // 長さ CHUNK_SIZE
    __global float* out_cos,         // 長さ CHUNK_SIZE
    __global float* out_native_sin,  // 長さ CHUNK_SIZE
    __global float* out_native_cos,  // 長さ CHUNK_SIZE
    const uint      chunk_hi         // 0..255
){
    const uint gid = (uint)get_global_id(0);  // 0 .. (CHUNK_SIZE-1)
    // ビットパターン（0x???????? の昇順）
    const uint u = (chunk_hi << CHUNK_LOG2) | gid;

    // ビットパターンを float に再解釈（非正規/NaN/Inf を含む）
    const float x = as_float(u);

    
    const double s  = sin((double)x);
    const double c  = cos((double)x);

    // 書き込み順は gid 昇順 = ビットパターン昇順
    out_sin[gid]        = (float)s;
    out_cos[gid]        = (float)c;
    
}
