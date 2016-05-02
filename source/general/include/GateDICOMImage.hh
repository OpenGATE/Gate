/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateDICOMImage
  \author Jérôme Suhard (jerome@suhard.fr)
          Thomas DESCHLER (thomas@deschler.fr)
  \date	April 2016
*/

#ifndef GATEDICOMIMAGE_HH
#define GATEDICOMIMAGE_HH

#include "itkImage.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"

#include <gdcmFilename.h>

#if ITK_VERSION_MAJOR >= 4
  #include "gdcmUIDGenerator.h"
#else
  #include "gdcm/src/gdcmFile.h"
  #include "gdcm/src/gdcmUtil.h"
#endif

#include "GateMiscFunctions.hh"

#include <string>


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
    double              GetPixelValue(const std::vector<int>);
    void                GetPixels( std::vector<int>&);
    unsigned int        GetPixelsCount();

  private:
    typedef signed short                        PixelType;
    typedef itk::Image< PixelType, 3 >          ImageType;
    typedef itk::ImageSeriesReader< ImageType > ReaderType;
    typedef itk::GDCMImageIO                    ImageIOType;
    typedef itk::GDCMSeriesFileNames            NamesGeneratorType;

    ReaderType::Pointer reader;
    std::vector<int> pixels;
    std::vector<int> vResolution;
    std::vector<double> vSpacing;
    std::vector<double> vSize;
    std::vector<double> vOrigin;
};

#endif
