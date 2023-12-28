#include <stdio.h>
#include <stdint.h>

typedef struct {
    char     chunkId[4];
    uint32_t chunkSize;
    char     format[4];
    char     subchunk1Id[4];
    uint32_t subchunk1Size;
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    char     subchunk2Id[4];
    uint32_t subchunk2Size;
} WavHeader;

int main() {
    FILE* file = fopen("yourfile.wav", "rb");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    WavHeader header;
    fread(&header, sizeof(WavHeader), 1, file);

    // Check if the file is a WAV file
    if (header.chunkId[0] != 'R' || header.chunkId[1] != 'I' || header.chunkId[2] != 'F' || header.chunkId[3] != 'F' ||
        header.format[0] != 'W' || header.format[1] != 'A' || header.format[2] != 'V' || header.format[3] != 'E') {
        fprintf(stderr, "Not a valid WAV file\n");
        fclose(file);
        return 1;
    }

    printf("Channels: %d\n", header.numChannels);
    printf("Sample Rate: %d Hz\n", header.sampleRate);
    printf("Bits per Sample: %d\n", header.bitsPerSample);

    // Calculate the number of samples
    uint32_t numSamples = header.subchunk2Size / (header.numChannels * (header.bitsPerSample / 8));

    // Read and process the audio data
    if (header.numChannels == 1) {
        // Mono audio
        int16_t* data = malloc(numSamples * sizeof(int16_t));
        fread(data, sizeof(int16_t), numSamples, file);
        // Process mono audio data here
        free(data);
    } else if (header.numChannels == 2) {
        // Stereo audio
        int16_t* data = malloc(numSamples * header.numChannels * sizeof(int16_t));
        fread(data, sizeof(int16_t), numSamples * header.numChannels, file);
        // Process stereo audio data here
        free(data);
    } else {
        fprintf(stderr, "Unsupported number of channels: %d\n", header.numChannels);
    }

    fclose(file);
    return 0;
}
