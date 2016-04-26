/*
 * This confidential and proprietary software may be used only as
 * authorised by a licensing agreement from ARM Limited
 *   (C) COPYRIGHT 2013 ARM Limited
 *       ALL RIGHTS RESERVED
 * The entire notice above must be reproduced on all authorised
 * copies and copies may only be made to the extent permitted
 * by a licensing agreement from ARM Limited.
 */

#include "image.h"
#include <fstream>
#include <iostream>

using namespace std;

bool saveToBitmap(string filename, int width, int height, const unsigned char* imageData)
{
    /* Try and open the file for writing. */
    fstream imageFile(filename.c_str(), ios::out);
    if(!imageFile.is_open())
    {
        cerr << "Unable to open " << filename << ". " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    /* Magic header bits come from the bitmap specification. */
    const struct bitmapMagic magic = { {0x42, 0x4d} };
    struct bitmapHeader header;
    struct bitmapInformationHeader informationHeader;

    /*
     * Each row of the data must be padded to a multiple of 4 bytes according to the bitmap specification.
     * This method uses three bytes per pixel (hence the width * 3).
     * Then we increase the padding until it is divisible by 4.
     */
    int paddedWidth = width * 3;
    while((paddedWidth % 4) != 0)
    {
        paddedWidth++;
    }

    /* Setup the bitmap header. */
    header.fileSize = sizeof(magic) + sizeof(header) + sizeof(informationHeader) + paddedWidth * height;
    header.creator1 = 0;
    header.creator2 = 0;
    header.offset = sizeof(magic) + sizeof(header) + sizeof(informationHeader);

    /* Setup the bitmap information header. */
    informationHeader.size = sizeof(informationHeader);
    informationHeader.width = width;
    informationHeader.height = height;
    informationHeader.numberOfColorPlanes = 1;
    informationHeader.bitsPerPixel = 24;
    informationHeader.compressionType = 0;
    informationHeader.rawBitmapSize = paddedWidth * height;
    informationHeader.horizontalResolution = 2835;
    informationHeader.verticalResolution = 2835;
    informationHeader.numberOfColors = 0;
    informationHeader.numberOfImportantColors = 0;

    /* Try to write the header data. */
    if(imageFile.write((char*)&magic, sizeof(magic)).bad() ||
       imageFile.write((char*)&header, sizeof(header)).bad() ||
       imageFile.write((char*)&informationHeader, sizeof(informationHeader)).bad())
    {
        cerr << "Failed to write bitmap header. " << __FILE__ << ":"<< __LINE__ << endl;
        if (imageFile.is_open())
        {
            imageFile.close();
        }
        return false;
    }

    for (int y = height - 1; y >= 0; y--)
    {
        for (int x = 0; x < width; x++)
        {
            /* The pixels lie in RGB order in memory, we need to store them in BGR order. */
            unsigned char rgb[3];
            rgb[2] = imageData[3 * (y * informationHeader.width + x) + 0];
            rgb[1] = imageData[3 * (y * informationHeader.width + x) + 1];
            rgb[0] = imageData[3 * (y * informationHeader.width + x) + 2];

            if(imageFile.write((char*)&rgb, 3).bad())
            {
                if (imageFile.is_open())
                {
                    imageFile.close();
                }
                return false;
            }
        }
        /*
         * At the end of the row, write out blank bytes to ensure the row length is
         * a multiple of 4 bytes (part of the bitmap specification).
         */
        for (int x = width * 3; x < paddedWidth; x++)
        {
            char b = 0;
            if(imageFile.write(&b, 1).bad())
            {
                if (imageFile.is_open())
                {
                    imageFile.close();
                }
                return false;
            }
        }
    }

    return true;
}

bool loadFromBitmap(const string filename, int* const width, int* const height, unsigned char **imageData)
{
     /* Try and open the file for reading. */
    ifstream imageFile(filename.c_str(), ios::in);
    if(!imageFile.is_open())
    {
        cerr << "Unable to open " << filename << ". " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    /*
     * Read and check the headers to make sure we support the type of bitmap passed in.
     */
    struct bitmapMagic magic;
    struct bitmapHeader header;
    struct bitmapInformationHeader informationHeader;

    if (imageFile.read((char*)&magic, sizeof(magic)).bad() || magic.magic[0] != 0x42 || magic.magic[1] != 0x4d)
    {
        /* Not a valid BMP file header */
        cerr << "Not a valid BMP file header. " << __FILE__ << ":"<< __LINE__ << endl;
        if (imageFile.is_open())
        {
            imageFile.close();
        }
        return false;
    }

    /* 54 is the standard size of a bitmap header. */
    if (imageFile.read((char*)&header, sizeof(header)).bad() || header.offset != 54)
    {
        /* Not a supported BMP format */
        cerr << "Not a supported BMP format. " << __FILE__ << ":"<< __LINE__ << endl;
        if (imageFile.is_open())
        {
            imageFile.close();
        }
        return false;
    }

    if (imageFile.read((char*)&informationHeader, sizeof(informationHeader)).bad() || informationHeader.compressionType != 0 || informationHeader.bitsPerPixel != 24)
    {
        /* We only support uncompressed 24-bits per pixel RGB */
        cerr << "We only support uncompressed 24-bits per pixel RGB. " << __FILE__ << ":"<< __LINE__ << endl;
        if (imageFile.is_open())
        {
            imageFile.close();
        }
        return false;
    }

    int row_delta;
    int first_row;
    int after_last_row;
    if (informationHeader.height > 0)
    {
        /* The image is stored upside down in memory */
        row_delta = -1;
        first_row = informationHeader.height - 1;
        after_last_row = -1;
    }
    else
    {
        informationHeader.height = -informationHeader.height;
        row_delta = 1;
        first_row = 0;
        after_last_row = informationHeader.height;
    }

    /* Calculate the paddle of the image to skip it when reading the buffer. */
    int paddedWidth = informationHeader.width * 3;
    while((paddedWidth % 4) != 0)
    {
        paddedWidth++;
    }

    /* 24-bits per pixel means 3 bytes of data per pixel. */
    int size = 3 * paddedWidth * informationHeader.height;
    *imageData = new unsigned char[size];
    unsigned char* readBuffer = new unsigned char[size];

    /* Try to read in the image data. */
    if (imageFile.read((char*)readBuffer, size).bad())
    {
        cerr << "Error reading main image data. " << __FILE__ << ":"<< __LINE__ << endl;
        if (imageFile.is_open())
        {
            imageFile.close();
        }
        if (readBuffer != NULL)
        {
            delete [] readBuffer;
        }
        return false;
    }

    int readBufferIndex = 0;
    /* Loop throught the image data and store it at the output data location. */
    for (int y = first_row; y != after_last_row; y += row_delta)
    {
        for (int x = 0; x < informationHeader.width; x++)
        {
            /* The pixels lie in BGR order, we need to resort them into RGB */
            (*imageData)[3 * (y * informationHeader.width + x) + 0] = readBuffer[readBufferIndex + 2];
            (*imageData)[3 * (y * informationHeader.width + x) + 1] = readBuffer[readBufferIndex + 1];
            (*imageData)[3 * (y * informationHeader.width + x) + 2] = readBuffer[readBufferIndex + 0];

            readBufferIndex += 3;
        }
        /* Skip padding. */
        readBufferIndex += paddedWidth - (informationHeader.width * 3);
    }

    *width  = informationHeader.width;
    *height = informationHeader.height;

    if (imageFile.is_open())
    {
        imageFile.close();
    }
    if (readBuffer != NULL)
    {
        delete [] readBuffer;
    }

    return true;
}

bool luminanceToRGB(const unsigned char* luminanceData, unsigned char* rgbData, int width, int height)
{
    if (luminanceData == NULL)
    {
        cerr << "luminanceData cannot be NULL. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    if (rgbData == NULL)
    {
        cerr << "rgbData cannot be NULL. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    for (int n = width * height - 1; n >= 0; --n)
    {
        unsigned char d = luminanceData[n];
        rgbData[3 * n + 0] = d;
        rgbData[3 * n + 1] = d;
        rgbData[3 * n + 2] = d;
    }
    return true;
}

bool RGBToLuminance(const unsigned char* const rgbData, unsigned char* const luminanceData, int width, int height)
{
    if (rgbData == NULL)
    {
        cerr << "rgbData cannot be NULL. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    if (luminanceData == NULL)
    {
        cerr << "luminanceData cannot be NULL. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    for (int n = width * height - 1; n >= 0; --n)
    {
        float r = rgbData[3 * n + 0];
        float g = rgbData[3 * n + 1];
        float b = rgbData[3 * n + 2];
        luminanceData[n] = (unsigned char) (0.2126f * r + 0.7152f * g + 0.0722f * b);
    }
    return true;
}

bool RGBToRGBA(const unsigned char* const rgbData, unsigned char* const rgbaData, int width, int height)
{
    if (rgbData == NULL)
    {
        cerr << "rgbData cannot be NULL. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    if (rgbaData == NULL)
    {
        cerr << "rgbaData cannot be NULL. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    for (int n = 0; n < width * height; n++)
    {
        /* Copy the RGB components directly. */
        rgbaData[4 * n + 0] = rgbData[3 * n + 0];
        rgbaData[4 * n + 1] = rgbData[3 * n + 1];
        rgbaData[4 * n + 2] = rgbData[3 * n + 2];

        /* Set the alpha channel to 255 (fully opaque). */
        rgbaData[4 * n + 3] = (unsigned char)255;
    }
    return true;
}

bool RGBAToRGB(const unsigned char* const rgbaData, unsigned char* const rgbData, int width, int height)
{
    if (rgbaData == NULL)
    {
        cerr << "rgbaData cannot be NULL. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    if (rgbData == NULL)
    {
        cerr << "rgbData cannot be NULL. " << __FILE__ << ":"<< __LINE__ << endl;
        return false;
    }

    for (int n = 0; n < width * height; n++)
    {
        /* Copy the RGB components but throw away the alpha channel. */
        rgbData[3 * n + 0] = rgbaData[4 * n + 0];
        rgbData[3 * n + 1] = rgbaData[4 * n + 1];
        rgbData[3 * n + 2] = rgbaData[4 * n + 2];
    }
    return true;
}
