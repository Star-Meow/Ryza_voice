#!/usr/bin/env python3
"""Parse Atelier Ryza 2 VOICE.ktsl2asbin -> per-cue metadata CSV.

For each cue: sound_id, flags, sample rate, num samples, and the offset/size
of its KOVS block inside VOICE.ktsl2stbin. Also reports whether a cue carries
a plaintext/encrypted NAME string (per vgmstream src/meta/ktsr.c).

KTSC container:
  0x00 'KTSC' 0x04 ver 0x08 cue_count 0x0c table1_off 0x10 table2_off
  table1: per-cue sound_id; table2: per-cue subentry offset
KTSR subentry:
  0x00 'KTSR' 0x04 audio_id ... chunks from 0x40
  chunk = magic(4 BE) + total_size(4 LE, includes header)
    0xBD888C36 config: +0x08 stream_id +0x0c flags +0x28 name_off
    0xC5CCCB70 sound:   +0x08 sound_id  +0x0c flags +0x10 stream_count
                        +0x14 hdr_ptr  +0x18 name_off
  sound header (external, e.g. type 0xDF92529F):
    +0x0c channels +0x14 sample_rate +0x18 num_samples
    +0x34 stbin_offset +0x38 stbin_size
"""
import struct
import sys
import csv

CHUNK_CONFIG = 0xBD888C36
CHUNK_SOUND = 0xC5CCCB70
EXTERNAL_TYPES = {0x38D0437D, 0x3DEA478D, 0xDF92529F, 0x6422007C,
                  0x793A1FD7, 0xA0F4FC6C}

def cstr(raw):
    i = raw.find(0)
    return raw[:i] if i >= 0 else raw

def parse(data):
    assert data[:4] == b'KTSC'
    cue_count = struct.unpack_from('<I', data, 0x08)[0]
    t1_off = struct.unpack_from('<I', data, 0x0c)[0]
    t2_off = struct.unpack_from('<I', data, 0x10)[0]

    out = []
    for n in range(cue_count):
        sound_id = struct.unpack_from('<I', data, t1_off + 4 * n)[0]
        sub_off = struct.unpack_from('<I', data, t2_off + 4 * n)[0]
        sub = data[sub_off:]
        assert sub[:4] == b'KTSR', f"cue {n}: no KTSR at {sub_off:#x}"
        audio_id = struct.unpack_from('<I', sub, 0x04)[0]

        sound_chunk = cfg_chunk = None
        p = 0x40
        while p + 8 <= len(sub):
            magic = struct.unpack_from('>I', sub, p)[0]
            size = struct.unpack_from('<I', sub, p + 4)[0]
            if size < 8 or p + size > len(sub):
                break
            if magic == CHUNK_SOUND and sound_chunk is None:
                sound_chunk = sub[p:p + size]
            elif magic == CHUNK_CONFIG and cfg_chunk is None:
                cfg_chunk = sub[p:p + size]
            p += size

        rec = {'cue': n, 'sub_off': sub_off, 'audio_id': audio_id,
               'sound_id': sound_id}

        # --- names ---
        cfg_name = snd_name = ''
        if cfg_chunk is not None:
            cfg_stream_id = struct.unpack_from('<I', cfg_chunk, 0x08)[0]
            cfg_flags = struct.unpack_from('<I', cfg_chunk, 0x0c)[0]
            no = struct.unpack_from('<I', cfg_chunk, 0x28)[0]
            if cfg_stream_id == sound_id and no > 0:
                raw = cstr(cfg_chunk[no:no + 256])
                cfg_name = raw.hex()
                rec['cfg_flags'] = cfg_flags
        if sound_chunk is not None:
            snd_flags = struct.unpack_from('<I', sound_chunk, 0x0c)[0]
            no = struct.unpack_from('<I', sound_chunk, 0x18)[0]
            if no > 0:
                raw = cstr(sound_chunk[no:no + 256])
                snd_name = raw.hex()
                rec['snd_flags'] = snd_flags
        rec['cfg_name_hex'] = cfg_name
        rec['snd_name_hex'] = snd_name

        # --- stream offset/size into stbin ---
        if sound_chunk is not None:
            hdr_ptr = struct.unpack_from('<I', sound_chunk, 0x14)[0]
            # header_offset = u32le(sound_chunk + hdr_ptr) + sound_chunk
            rel = struct.unpack_from('<I', sound_chunk, hdr_ptr)[0]
            # `rel` is relative to the sound chunk start; find that in `sub`
            chunk_start = sub.find(sound_chunk[:8], 0x40)
            hdr = chunk_start + rel
            stype = struct.unpack_from('>I', sub, hdr)[0]
            rec['hdr_type'] = f"{stype:#010x}"
            if stype in EXTERNAL_TYPES:
                rec['channels'] = struct.unpack_from('<I', sub, hdr + 0x0c)[0]
                rec['codec'] = struct.unpack_from('<I', sub, hdr + 0x14)[0]
                rec['sample_rate'] = struct.unpack_from('<I', sub, hdr + 0x18)[0]
                rec['num_samples'] = struct.unpack_from('<I', sub, hdr + 0x1c)[0]
                rec['loop_start'] = struct.unpack_from('<i', sub, hdr + 0x24)[0]
                if stype == 0x3DEA478D:  # Nioh PC variant
                    rec['stbin_off'] = struct.unpack_from('<I', sub, hdr + 0x30)[0]
                    rec['stbin_size'] = struct.unpack_from('<I', sub, hdr + 0x34)[0]
                else:
                    rec['stbin_off'] = struct.unpack_from('<I', sub, hdr + 0x34)[0]
                    rec['stbin_size'] = struct.unpack_from('<I', sub, hdr + 0x38)[0]
        out.append(rec)
    return out

if __name__ == '__main__':
    asbin = sys.argv[1] if len(sys.argv) > 1 else \
        r'D:\SteamLibrary\steamapps\common\Atelier Ryza 2\Data\x64\Sound\VOICE.ktsl2asbin'
    out_csv = sys.argv[2] if len(sys.argv) > 2 else 'voice_cues.csv'
    data = open(asbin, 'rb').read()
    recs = parse(data)
    keys = ['cue', 'sound_id', 'audio_id', 'hdr_type', 'channels', 'codec',
            'sample_rate', 'num_samples', 'loop_start', 'stbin_off',
            'stbin_size', 'cfg_name_hex', 'snd_name_hex', 'sub_off']
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k, '') for k in keys})
    named = sum(1 for r in recs if r.get('cfg_name_hex') or r.get('snd_name_hex'))
    print(f"cues: {len(recs)}  with name bytes: {named}")
    print(f"wrote {out_csv}")
    for r in recs[:3]:
        print(r)
