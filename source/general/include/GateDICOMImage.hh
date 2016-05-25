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
#include "itkImageFileWriter.h"
//#include <itkDCMTKImageIO.h>

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
    void ReadSeries(const std::string, const std::string);
    std::vector<int> GetResolution();
    std::vector<double> GetSpacing();
    std::vector<double> GetOrigin();
    std::vector<double> GetImageSize();
    void SetResolution(std::vector<long unsigned int>);
    void SetSpacing(std::vector<double>);
    void SetOrigin(std::vector<double>);
    void dumpIO();
    void Write(const std::string);

    template<class PixelType>
    void GetPixels(std::vector<PixelType>& data);

    template<class PixelType>
    void SetPixels(std::vector<PixelType>& data);

  private:
    typedef itk::Image<signed short,3>          ImageType;
    typedef itk::ImageSeriesReader< ImageType > ReaderType;
    typedef itk::GDCMImageIO                    ImageIOType;
    typedef itk::GDCMSeriesFileNames            NamesGeneratorType;
    //typedef itk::DCMTKImageIO                   DCMTKIOType;

    ReaderType::Pointer reader;

    ImageType::Pointer dicomIO;

    unsigned int pixelsCount;

    std::vector<int> vResolution;
    std::vector<double> vSpacing;
    std::vector<double> vSize;
    std::vector<double> vOrigin;
};

#include "GateDICOMImage.icc"

#endif
