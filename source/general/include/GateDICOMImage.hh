/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*
  \class  GateDICOMImage
  \author Thomas DESCHLER (thomas@deschler.fr)
  based on the work of
          Jérôme Suhard (jerome@suhard.fr)
          Albertine Dubois (adubois@imnc.in2p3.fr)
  \date	April 2016
*/
#include "GateConfiguration.h"
#ifdef  GATE_USE_ITK

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
    std::vector<long unsigned int> GetResolution();
    std::vector<double> GetSpacing();
    std::vector<double> GetOrigin();
    std::vector<double> GetSize();
    G4RotationMatrix GetRotationMatrix();
    void SetResolution(std::vector<long unsigned int>);
    void SetSpacing(std::vector<double>);
    void SetOrigin(std::vector<double>);
    void dumpIO();
    void Write(const std::string);
    double round_to_digits(double,int);

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

    ImageType::Pointer image;

    ImageType::Pointer dicomIO;

    unsigned int pixelsCount;

    std::vector<long unsigned int> vResolution;
    std::vector<double> vSpacing;
    std::vector<double> vSize;
    std::vector<double> vOrigin;

    G4RotationMatrix rotationMatrix;
};

#include "GateDICOMImage.icc"

#endif
#endif
