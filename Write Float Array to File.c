#include <stdio.h>

int main() {
    float floatArray[] = {1.0f, 2.5f, 3.7f, 4.2f, 5.8f};
    size_t array_size = sizeof(floatArray) / sizeof(floatArray[0]);

    FILE* file = fopen("data_float.bin", "wb");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    fwrite(floatArray, sizeof(float), array_size, file);
    fclose(file);

    return 0;
}
