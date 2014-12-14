#include <iostream>
#include <cassert>
#include <cstring>
#include <vector>
#include <ctime>
#include <fstream>

#include <Eigen/Core>

#include "ImageRGB.hpp"
#include "ioJPG.hpp"
#include "exif.h"
#include "HdrCompute.hpp"
#include <limits>
#include <string>

using namespace kn;


//////////////////////////////////////////////////////////////////////////////////////////////
// open the file "filename" and copy the file content into a string (required for exif reader)
std::string fileToString(const std::string& filename)
{
    std::ifstream file(filename.c_str());//, std::ios::binary);
    if (!file) return "";
    std::string str(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
    return str;
}


//////////////////////////////////////////////////////////////////////////////////////////////
void drawPixels(ImageRGB8u &image, const std::vector<Eigen::Vector2i> &pixels){
    for(uint i=0; i<pixels.size(); ++i){
        image(pixels[i][0],pixels[i][1])[0] = 255;
        image(pixels[i][0],pixels[i][1])[1] = 0;
        image(pixels[i][0],pixels[i][1])[2] = 0;
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////
void exifParsingError(const int parseSuccess){
    switch(parseSuccess){
    case 1982 : // No JPEG markers found in buffer, possibly invalid JPEG file
        std::cout << "exif parsing error : PARSE_EXIF_ERROR_NO_JPEG" << std::endl;
        break;
    case 1983 : // No EXIF header found in JPEG file.
        std::cout << "exif parsing error : PARSE_EXIF_ERROR_NO_EXIF" << std::endl;
        break;
    case 1984 : // Byte alignment specified in EXIF file was unknown (not Motorola or Intel).
        std::cout << "exif parsing error : PARSE_EXIF_ERROR_UNKNOWN_BYTEALIGN" << std::endl;
        break;
    case 1985 : // EXIF header was found, but data was corrupted.
        std::cout << "exif parsing error : PARSE_EXIF_ERROR_CORRUPT" << std::endl;
        break;
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////
void loadImages(const int argc, char **argv, std::vector<ImageRGB8u> &images, std::vector<double> &exposure){

    for(int i=1; i<argc; ++i){

        //load the image
        ImageRGB8u imgageTmp;
        std::cout << "loading '" << argv[i] << "' ...";
        loadJPG(imgageTmp,argv[i]);
        images.push_back(imgageTmp);
        std::cout << " done" << std::endl;

        // load the exposure duration from the exif
        EXIFInfo exifReader;
        int parseSuccess = exifReader.parseFrom(fileToString(argv[i]));
        if(parseSuccess != PARSE_EXIF_SUCCESS){
            exifParsingError(parseSuccess);
            exit(0);
        }
        // std::cout << "   wxh       : " << exifReader.ImageWidth << " x " << exifReader.ImageHeight << std::endl;
        // std::cout << "   exposure  : " << exifReader.ExposureTime << " s" << std::endl;
        // std::cout << "   flash     : " << ((exifReader.Flash==0)?"no":"yes") << std::endl;
        // std::cout << "   camera    : " << exifReader.Model << std::endl;
        // std::cout << "   ISO       : " << exifReader.ISOSpeedRatings << std::endl;
        // std::cout << "   apperture : " << exifReader.FNumber << std::endl;
        //std::cout << std::endl;

        // update exposure
        exposure.push_back((double)exifReader.ExposureTime);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    Eigen::setNbThreads(4);
    // check arguments
    if(argc < 3){
        std::cerr << "usage : " << argv[0] << " image_1.jpg ... image_n.jpg" << std::endl;
        std::cerr << "or    : " << argv[0] << " dirname/*.jpg" << std::endl;
        exit(0);
    }

    // load images and exposure time
    std::vector<ImageRGB8u> images;
    std::vector<double> exposure;
    loadImages(argc, argv, images, exposure);

    std::vector<Eigen::MatrixXi> imagesMatrixGray, imagesMatrixR, imagesMatrixG, imagesMatrixB;
    std::vector<std::vector<Eigen::MatrixXi>> imagesMatricesColors,imagesMatrixGrayEncapsulate;

    uint widthImage = images[0].width();
    uint heightImage = images[0].height();

    //tranform images to eigen matrix
    for(unsigned i = 0; i < images.size(); ++i) {
        Eigen::MatrixXi tmpG(heightImage, widthImage);
        Eigen::MatrixXi tmpR(heightImage, widthImage);
        Eigen::MatrixXi tmpGreen(heightImage, widthImage);
        Eigen::MatrixXi tmpB(heightImage, widthImage);

        HdrCompute::transformImageToMatrixGray(images[i], tmpG);
        HdrCompute::transformImageToMatrix(images[i], tmpR, tmpGreen, tmpB);

        imagesMatrixGray.push_back(std::move(tmpG));
        imagesMatrixR.push_back(std::move(tmpR));
        imagesMatrixG.push_back(std::move(tmpGreen));
        imagesMatrixB.push_back(std::move(tmpB));
    }

    //push into vector of vector<matrix> for loop: simplify argument function
    imagesMatrixGrayEncapsulate.push_back(std::move(imagesMatrixGray)); // respect generectite compareto RGB
    imagesMatricesColors.push_back(std::move(imagesMatrixR));
    imagesMatricesColors.push_back(std::move(imagesMatrixG));
    imagesMatricesColors.push_back(std::move(imagesMatrixB));

    //select pixels for least squares
    std::vector<Eigen::Vector2i> pixels;
    for(uint x=heightImage/2-50; x<heightImage/2 - 50; ++x){
        pixels.push_back(Eigen::Vector2i(widthImage/2,x));
    }


    ImageRGB8u final(widthImage, heightImage);

    // Choices: expClamp, reinhard or linear
    // HdrCompute::handleGray(final, pixels, imagesMatrixGrayEncapsulate, exposure, 0,255);
    HdrCompute::handleRGB("expClamp", final, pixels, imagesMatricesColors, exposure,0,255);

    saveJPG(final,"../output/internet-3000.jpg");

    return 0;
}



