#!/usr/bin/env python3
"""De-obfuscate a KOVS block from VOICE.ktsl2stbin into a plain .ogg file.

KOVS block layout (vgmstream src/meta/ogg_vorbis.c, kovs_ogg_decryption_callback):
  0x00 'KOVS'  0x04 block_size  0x08 loop_start  0x0c 4  0x10 channels
  0x20 ogg stream start
The first 0x100 bytes of the Ogg stream are XOR'd with their own offset:
  buf[i] ^= (i & 0xFF);  (i counted from the start of the Ogg stream)
The rest is plain Ogg Vorbis.

Usage: python kovs_to_ogg.py <stbin> <offset> <size> <out.ogg>
       python kovs_to_ogg.py --cue N   # use voice_cues.csv entry N
"""
import struct
import sys
import csv
import os

STBIN = r'D:\SteamLibrary\steamapps\common\Atelier Ryza 2\Data\x64\Sound\VOICE.ktsl2stbin'
XOR_LEN = 0x100


def deobfuscate(block):
    """block = raw KOVS bytes (including 0x20 header) -> plain ogg bytes."""
    assert block[:4] == b'KOVS', f"not a KOVS block: {block[:4]!r}"
    ogg = bytearray(block[0x20:])
    for i in range(min(XOR_LEN, len(ogg))):
        ogg[i] ^= i & 0xFF
    return bytes(ogg)


def main():
    args = sys.argv[1:]
    if args and args[0] == '--cue':
        n = int(args[1])
        row = None
        with open(os.path.join(os.path.dirname(__file__), '..', 'voice_index.csv'),
                  encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if int(r['cue_index']) == n:
                    row = r
                    break
        assert row, f"cue {n} not found"
        off, size = int(row['stbin_offset']), int(row['stbin_size'])
        out = args[2] if len(args) > 2 else f"cue_{n:05d}.ogg"
    else:
        off, size, out = int(args[0], 0), int(args[1], 0), args[2]

    with open(STBIN, 'rb') as f:
        f.seek(off)
        block = f.read(size)
    ogg = deobfuscate(block)
    with open(out, 'wb') as f:
        f.write(ogg)
    print(f"KOVS@{off:#x} size={size} -> {out} ({len(ogg)} bytes)")
    print(f"  head: {ogg[:4]!r} (expect b'OggS')")


if __name__ == '__main__':
    main()
