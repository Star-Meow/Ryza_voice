#!/usr/bin/env python3
"""Extract named cue lists from bare-KTSR sound banks (e.g. battle voices).

Bare KTSR (.ktsl2asbin that starts with 'KTSR'):
  0x0b platform  0x0c audio_id (LCG name-decryption seed)
  0x40 chunks: magic(4 BE) + total_size(4 LE, includes 8-byte header)
    0xBD888C36 config: +0x08 stream_id +0x0c flags +0x28 name_off (encrypted if flags&0x0200)
    0xC5CCCB70 sound:   +0x08 sound_id  +0x0c flags +0x18 name_off (encrypted if flags&0x0008)
Names are LCG-XOR encrypted (classic rand: seed = 0x343FD*seed + 0x269EC3, XOR (seed>>16)&0xFF).
"""
import struct
import sys
import os
import csv

CHUNK_CONFIG = 0xBD888C36
CHUNK_SOUND = 0xC5CCCB70

def lcg(s):
    return (0x343FD * s + 0x269EC3) & 0xFFFFFFFF

def decrypt_name(raw, seed):
    out = bytearray(raw)
    s = seed
    for i in range(len(out)):
        if out[i] == 0:
            break
        s = lcg(s)
        out[i] ^= (s >> 16) & 0xFF
        if out[i] == 0:
            break
    return bytes(out)

def cstr(b):
    i = b.find(0)
    return b[:i] if i >= 0 else b

def parse_bare(data):
    assert data[:4] == b'KTSR'
    audio_id = struct.unpack_from('<I', data, 0x0c)[0]
    entries = {}  # stream_id -> {'sound':.., 'cfg':..}
    order = []
    p = 0x40
    while p + 8 <= len(data):
        magic = struct.unpack_from('>I', data, p)[0]
        size = struct.unpack_from('<I', data, p + 4)[0]
        if size < 8 or p + size > len(data):
            break
        c = data[p:p + size]
        if magic in (CHUNK_CONFIG, CHUNK_SOUND):
            sid = struct.unpack_from('<I', c, 0x08)[0]
            if sid not in entries:
                entries[sid] = {}
                order.append(sid)
            if magic == CHUNK_SOUND:
                no = struct.unpack_from('<I', c, 0x18)[0]
                flags = struct.unpack_from('<I', c, 0x0c)[0]
                if no > 0:
                    raw = cstr(c[no:no + 255])
                    if flags & 0x0008:
                        raw = decrypt_name(raw, audio_id)
                    entries[sid]['sound_name'] = raw
                entries[sid]['sound_flags'] = flags
            else:
                no = struct.unpack_from('<I', c, 0x28)[0]
                flags = struct.unpack_from('<I', c, 0x0c)[0]
                if no > 0:
                    raw = cstr(c[no:no + 255])
                    if flags & 0x0200:
                        raw = decrypt_name(raw, audio_id)
                    entries[sid]['cfg_name'] = raw
                entries[sid]['cfg_flags'] = flags
        p += size
    return audio_id, order, entries

if __name__ == '__main__':
    SND = r'D:\SteamLibrary\steamapps\common\Atelier Ryza 2\Data\x64\Sound'
    banks = sys.argv[1:] or ['014_battle_SP_character', '015_battle_character',
                             '012_battle_ally']
    for bank in banks:
        path = os.path.join(SND, bank + '.ktsl2asbin')
        data = open(path, 'rb').read()
        audio_id, order, entries = parse_bare(data)
        named = sum(1 for s in order if entries[s].get('cfg_name') or entries[s].get('sound_name'))
        print(f"\n=== {bank}: {len(order)} sounds, {named} named, audio_id={audio_id:#x}")
        for sid in order[:12]:
            e = entries[sid]
            print(f"  {sid:#010x} cfg={e.get('cfg_name', b'')!r} snd={e.get('sound_name', b'')!r}")
