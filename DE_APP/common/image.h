/*
 * This confidential and proprietary software may be used only as
 * authorised by a licensing agreement from ARM Limited
 *   (C) COPYRIGHT 2013 ARM Limited
 *       ALL RIGHTS RESERVED
 * The entire notice above must be reproduced on all authorised
 * copies and copies may only be made to the extent permitted
 * by a licensing agreement from ARM Limited.
 */

#ifndef IMAGE_H
#define IMAGE_H

#include <CL/cl.h>
#include <string>

/**
 * \file image.h
 * \brief Functions for working with bitmap images.
 */

/**
 * \brief Bitmap magic file header numbers.
 * \details See the BMP file format specification for more details.
 */
struct bitmapMagic
{
  unsigned char magic[2];
};

/**
 * \brief Bitmap header.
 * \details See the BMP file format specification for more details.
 */
struct bitmapHeader
{
  uint32_t fileSize; /**< \brief Total size of the bitmap in bytes. */
  uint16_t creator1; /**< \brief Reserved field which can be application defined. */
  uint16_t creator2; /**< \brief Reserved field which can be application defined. */
  uint32_t offset; /**< \brief Offset in bytes to the beginning of the image data block. */
};

/**
 * \brief Bitmap information header.
 * \details See the BMP file format specification for more details.
 */
struct bitmapInformationHeader
{
  uint32_t size; /**< Size of the information headers in bytes. */
  int32_t width; /**< Width of the image. */
  int32_t height; /**< Height of the image. */
  uint16_t numberOfColorPlanes; /**< The number of colour planes. The only legal value is 1. */
  uint16_t bitsPerPixel; /**< Number of bits per pixel in the image. */
  uint32_t compressionType; /**< Compression type. Use 0 for uncompressed. */
  uint32_t rawBitmapSize; /**< Size of the image data including padding (does not include the size of the headers). */
  int32_t horizontalResolution; /**< Resolution is in pixels per meter. */
  int32_t verticalResolution; /**< Resolution is in pixels per meter. */
  uint32_t numberOfColors; /**< Number of colours in the image, can be left as 0. */
  uint32_t numberOfImportantColors; /**< Generally ignored by applications. */
};

/**
 * \brief Save data as a bitmap image.
 * \details Save a block of 24-bits per pixel RGB image data out as a bitmap image.
 *          Output bitmap is uncompressed.
 * \param[in] filename The filename to use for the bitmap (should typically have the extension .bmp).
 * \param[in] width The width of the image to save (in pixels).
 * \param[in] height The height of the image to save (in pixels).
 * \param[in] imageData Pointer to the data block to save. Data must be 8-bits per component RGB and be in row-major format.
 *                      The size of the data block must be 3 * width * height bytes.
 * \return False if an error occurred, true otherwise.
 */
bool saveToBitmap(std::string filename, int width, int height, const unsigned char* imageData);

/**
 * \brief Load data from a bitmap image.
 * \details Load a block of 24-bits per pixel RGB image data from a bitmap image.
 *          Only supports uncompressed bitmaps.
 * \param[in] filename The filename of the bitmap to load.
 * \param[out] width Pointer to where the width of the image (in pixels) will be stored.
 * \param[out] height Pointer to where the height of the image (in pixels) will be stored.
 * \param[out] imageData Pointer to the data block loaded. Data is loaded as 8-bits per component RGB and in row-major format.
 *                       The size of the data block is 3 * width * height bytes. Data must be deleted by the calling application.
 * \return False if an error occurred, true otherwise.
 */
bool loadFromBitmap(std::string filename, int* width, int* height, unsigned char **imageData);

/**
 * \brief Convert 8-bits per pixel luminance data to 24-bits per pixel RGB data.
 * \details Each RGB pixel is created using the luminance value for each component.
 *          For example, a pixel with luminance of 125 will convert into an RGB pixel with values R = 125, G = 125, and B = 125.
 * \param[in] luminanceData Pointer to a block of 8-bits per pixel luminance data. Must be width * height bytes in size.
 * \param[out] rgbData Pointer to a data block containing the 24-bits per pixel RGB data.
 *                     The data block must be initialised with a size of 3 * width * height bytes.
 * \param[in] width The width of the image.
 * \param[in] height The height of the image.
 * \return False if an error occurred, true otherwise.
 */
bool luminanceToRGB(const unsigned char* luminanceData, unsigned char* rgbData, int width, int height);

/**
 * \brief Convert 24-bits per pixel RGB data to 8-bits per pixel luminance data.
 * \details Each luminance pixel is created using a weighted sum of the RGB values.
 *          The weightings are 0.2126R, 0.7152G, and 0.0722B.
 * \param[in] rgbData Pointer to a block of 24-bits per pixel RGB data. Must be 3 * width * height bytes in size.
 * \param[out] luminanceData Pointer to a data block containing the 8-bits per pixel luminanceData data.
 *                           The data block must be initialised with a size of width * height bytes.
 * \param[in] width The width of the image.
 * \param[in] height The height of the image.
 * \return False if an error occurred, true otherwise.
 */
bool RGBToLuminance(const unsigned char* rgbData, unsigned char* luminanceData, int width, int height);


/**
 * \brief Convert 24-bits per pixel RGB data to 32-bits per pixel RGBA data.
 * \details The alpha values are all set to 255.
 * \param[in] rgbData Pointer to a block of 24-bits per pixel RGB data. Must be 3 * width * height bytes in size.
 * \param[out] rgbaData Pointer to a data block containing the 32-bits per pixel RGBA data.
 *                      The data block must be initialised with a size of 4 * width * height bytes.
 * \param[in] width The width of the image.
 * \param[in] height The height of the image.
 * \return False if an error occurred, true otherwise.
 */
bool RGBToRGBA(const unsigned char* rgbData, unsigned char* rgbaData, int width, int height);

/**
 * \brief Convert 32-bits per pixel RGBA data to 24-bits per pixel RGB data.
 * \details The alpha values are discarded.
 * \param[in] rgbaData Pointer to a block of 32-bits per pixel RGBA data. Must be 4 * width * height bytes in size.
 * \param[out] rgbData Pointer to a data block containing the 24-bits per pixel RGB data.
 *                     The data block must be initialised with a size of 3 * width * height bytes.
 * \param[in] width The width of the image.
 * \param[in] height The height of the image.
 * \return False if an error occurred, true otherwise.
 */
bool RGBAToRGB(const unsigned char* rgbaData, unsigned char* rgbData, int width, int height);

#endif
