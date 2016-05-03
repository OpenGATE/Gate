/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateDICOMImage
  \author Thomas DESCHLER (thomas@deschler.fr)
  based on the work of
          Jérôme Suhard (jerome@suhard.fr)
  \date	April 2016
*/

#ifndef GATEDICOMIMAGE_HH
#define GATEDICOMIMAGE_HH

#include "itkImage.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"

#include <gdcmFilename.h>

#include "GateMiscFunctions.hh"

#include <string>

template<class PixelType> class GateImageT;

class GateDICOMImage
{
  //const unsigned int Dimension = 3;

  public:
    GateDICOMImage();
    ~GateDICOMImage() {}

    void Read(const std::string);
    void ReadSeries(const std::string, std::string="");
    std::vector<int> GetResolution();
    std::vector<double> GetSpacing();
    std::vector<double> GetOrigin();
    std::vector<double> GetImageSize();
    unsigned int        GetPixelsCount();
    int GetPixelValue(const std::vector<int>);

    template<class PixelType>
    void GetPixels(std::vector<PixelType> & data);

  private:
    typedef signed short                        TypeOfPixel;
    typedef itk::Image<TypeOfPixel, 3 >         ImageType;
    typedef itk::ImageSeriesReader< ImageType > ReaderType;
    typedef itk::GDCMImageIO                    ImageIOType;
    typedef itk::GDCMSeriesFileNames            NamesGeneratorType;

    ReaderType::Pointer reader;

    unsigned int pixelsCount;

    std::vector<int> vResolution;
    std::vector<double> vSpacing;
    std::vector<double> vSize;
    std::vector<double> vOrigin;
};

#include "GateDICOMImage.icc"

#endif
