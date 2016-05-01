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

#include "GateDICOMImage.hh"

//#if ITK_VERSION_MAJOR >= 4
//#if ( ( ITK_VERSION_MAJOR == 4 ) && ( ITK_VERSION_MINOR < 6 ) )

// INFO: reader->GetOutput() = http://www.itk.org/Doxygen/html/classitk_1_1GDCMImageIO.html


//-----------------------------------------------------------------------------
void GateDICOMImage::Read(const std::string fileName)
{
  reader = ReaderType::New();
  reader->SetFileName(fileName);
  reader->SetImageIO(ImageIOType::New());

  try
    {
    reader->Update();
    }
  catch (itk::ExceptionObject & e)
    {
    std::cerr << "exception in file reader " << std::endl;
    std::cerr << e << std::endl;
    exit(EXIT_FAILURE);
    }

  //reader->GetOutput()->GetUIDPrefix(); //FIXME
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDICOMImage::ReadSeries(const std::string seriesDirectory)
{
  reader = ReaderType::New();
  reader->SetImageIO(ImageIOType::New());

  NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();
  nameGenerator->SetUseSeriesDetails( true );
  nameGenerator->SetDirectory( seriesDirectory );

  if(nameGenerator->GetSeriesUIDs().size()==1)
  {
    std::cerr << "The folder " << seriesDirectory << " does not contain any DICOM series." << std::endl;
    exit(EXIT_FAILURE);
  }
  else if(nameGenerator->GetSeriesUIDs().size()>1)
  {
    std::cerr << "The folder " << seriesDirectory << " contains two or more DICOM series." << std::endl;
    exit(EXIT_FAILURE);
  }
  else
    reader->SetFileNames( nameGenerator->GetFileNames( nameGenerator->GetSeriesUIDs().begin()->c_str() ) );

  try
  {
    reader->Update();
  }
  catch (itk::ExceptionObject &excp)
  {
    std::cerr << "Exception thrown while reading the series" << std::endl;
    std::cerr << excp << std::endl;
    exit(EXIT_FAILURE);
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
std::vector<int> GateDICOMImage::GetResolution()
{
  ImageType::SizeType resolution = reader->GetOutput()->GetLargestPossibleRegion().GetSize();

  std::vector<int> vResolution(resolution.GetSizeDimension(),-1);
  for(size_t i=0;i<resolution.GetSizeDimension();i++)
    vResolution[i]=resolution[i];

  return vResolution;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
std::vector<double> GateDICOMImage::GetSpacing()
{
  ImageType::SpacingType spacing = reader->GetOutput()->GetSpacing();

  std::vector<double> vSpacing(reader->GetOutput()->GetImageDimension(),0.);
  for(size_t i=0;i<reader->GetOutput()->GetImageDimension();i++)
    vSpacing[i]=spacing[i];

  return vSpacing;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
std::vector<double> GateDICOMImage::GetImageSize()
{
  std::vector<double> vSize(reader->GetOutput()->GetImageDimension(),0.);
  for(size_t i=0;i<reader->GetOutput()->GetImageDimension();i++)
    vSize[i]=GetResolution()[i]*GetSpacing()[i];

  //std::cout << vSize[0] <<","<< vSize[1] <<","<< vSize[2] << " mm" << std::endl;

  return vSize;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
std::vector<double> GateDICOMImage::GetOrigin()
{
  ImageType::PointType origin = reader->GetOutput()->GetOrigin();

  std::vector<double> vOrigin(reader->GetOutput()->GetImageDimension(),0.);
  for(size_t i=0;i<reader->GetOutput()->GetImageDimension();i++)
    vOrigin[i]=origin[i];

  return vOrigin;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
double GateDICOMImage::GetPixelValue(const std::vector<int> coord)
{
  ImageType::IndexType index;
  if(coord.size()==reader->GetOutput()->GetImageDimension())
  {
    for(size_t i=0;i<reader->GetOutput()->GetImageDimension();i++)
      index[i]=coord[i];
  }

  return reader->GetOutput()->GetPixel(index);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDICOMImage::GetPixels(std::vector<int>& output)
{
  // Voir GateVDICOM.cc
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
unsigned int GateDICOMImage::GetPixelsCount()
{
  return GetResolution()[0]*GetResolution()[1]*GetResolution()[2];
}
//-----------------------------------------------------------------------------
