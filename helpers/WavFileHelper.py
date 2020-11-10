import struct

class WavFileHelper():
    
    def read_file_properties(self, filename):
        #開檔
        wave_file = open(filename,"rb")
        
        #先讀去前面不會用到的12位元
        riff = wave_file.read(12)
        #將剩下的36位元讀進來
        fmt = wave_file.read(36)
        #NumChannels在22到24位元
        num_channels_string = fmt[10:12]
        #'H'代表unsigned short
        num_channels = struct.unpack('<H', num_channels_string)[0]
        #SampleRate在24到28位元
        #'I'代表unsigned int
        sample_rate_string = fmt[12:16]
        sample_rate = struct.unpack("<I",sample_rate_string)[0]
        #BitsPerSample在34到36位元
        bit_depth_string = fmt[22:24]
        bit_depth = struct.unpack("<H",bit_depth_string)[0]

        return (num_channels, sample_rate, bit_depth)
