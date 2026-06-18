import OpenEXR
import Imath
import numpy as np
import sys

def add_exr(path_a, path_b, path_out):
    file_a = OpenEXR.InputFile(path_a)
    file_b = OpenEXR.InputFile(path_b)

    header_a = file_a.header()
    dw = header_a['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = ['R', 'G', 'B']

    result = {}
    for ch in channels:
        buf_a = np.frombuffer(file_a.channel(ch, pt), dtype=np.float32)
        buf_b = np.frombuffer(file_b.channel(ch, pt), dtype=np.float32)
        result[ch] = (buf_a + buf_b).tobytes()

    header_out = OpenEXR.Header(width, height)
    header_out['compression'] = header_a.get('compression', Imath.Compression(Imath.Compression.PIZ_COMPRESSION))

    out = OpenEXR.OutputFile(path_out, header_out)
    out.writePixels(result)
    out.close()

    file_a.close()
    file_b.close()
    print(f"Written: {path_out}")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <image_a.exr> <image_b.exr> <output.exr>")
        sys.exit(1)
    add_exr(sys.argv[1], sys.argv[2], sys.argv[3])
