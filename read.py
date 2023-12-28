import struct

file_path = 'data_float.bin'

# Open the file in binary mode and read the data using struct.unpack
with open(file_path, 'rb') as file:
    # 'f' represents the float format
    data = struct.unpack('f' * (file.seek(0, 2) // struct.calcsize('f')), file.read())

print("Read data in Python:", data)
