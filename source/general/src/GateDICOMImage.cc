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
// INFO: NamesGeneratorType::Pointer = http://www.itk.org/Doxygen/html/classitk_1_1GDCMSeriesFileNames.html


//-----------------------------------------------------------------------------
GateDICOMImage::GateDICOMImage()
{
  vResolution.resize(0);
  vSpacing.resize(0);
  vSize.resize(0);
  vOrigin.resize(0);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDICOMImage::Read(const std::string fileName)
{
  std::string path = gdcm::Filename(fileName.c_str()).GetPath();

  reader = ReaderType::New();
  reader->SetFileName(fileName);

  ImageIOType::Pointer imgIO = ImageIOType::New();
  reader->SetImageIO(imgIO);

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

  #if ITK_VERSION_MAJOR >= 4
    std::string seriesUID = gdcm::UIDGenerator().Generate();
  #else
    std::string seriesUID = gdcm::Util::CreateUniqueUID( gdcmIO->GetUIDPrefix());
  #endif

  ReaderType::DictionaryRawPointer dict = (*(reader->GetMetaDataDictionaryArray()))[0];
  typedef itk::MetaDataObject< std::string >  MetaDataStringType;
  MetaDataStringType::Pointer entryvalue =
    dynamic_cast<MetaDataStringType *>( dict->Find( "0020|000e" )->second.GetPointer() );//OK Mais incomplet

  if( entryvalue )
    seriesUID=entryvalue->GetMetaDataObjectValue();
  else
    GateError("Can't find the series UID from the file" << Gateendl);

  GateMessage("Image", 5, "GateDICOMImage::Read: File: " << fileName <<", series UID: "<< seriesUID << Gateendl);

  ReadSeries(path,seriesUID);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDICOMImage::ReadSeries(const std::string seriesDirectory, std::string UID)
{
  GateMessage("Image", 5, "GateDICOMImage::ReadSeries: series path: " << seriesDirectory <<", series UID: "<< UID << Gateendl);

  reader = ReaderType::New();
  reader->SetImageIO(ImageIOType::New());

  NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();
  nameGenerator->SetUseSeriesDetails( true );
  nameGenerator->SetDirectory( seriesDirectory );

    std::cout << std::endl << "The directory " << seriesDirectory << std::endl << std::endl;
    std::cout << "Contains " << nameGenerator->GetSeriesUIDs().size() << " DICOM Series:";
    std::cout << std::endl << std::endl;

    std::vector< std::string >::const_iterator seriesItr = nameGenerator->GetSeriesUIDs().begin();
    std::vector< std::string >::const_iterator seriesEnd = nameGenerator->GetSeriesUIDs().end();

    while( seriesItr != seriesEnd )
    {
      std::string seriesUID(seriesItr->c_str());
      if(UID!="" && UID==seriesUID.substr(0, UID.size()))
        UID=seriesItr->c_str();
      //std::cout << "UID: " << UID << "seriesUID.substr: " << seriesUID.substr(0, UID.size()) << std::endl;
      std::cout << seriesItr->c_str() << std::endl;
      ++seriesItr;
    }

  if(UID!="")
    reader->SetFileNames(nameGenerator->GetFileNames(UID));
  else if(nameGenerator->GetSeriesUIDs().size()==1)
  {
    GateError( "The folder " << seriesDirectory << " does not contain any DICOM series." << Gateendl);
    exit(EXIT_FAILURE);
  }
  else if(nameGenerator->GetSeriesUIDs().size()>1)
  {
    GateError( "The folder " << seriesDirectory << " contains two or more DICOM series." << Gateendl);
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
  if(vResolution.size()==0)
  {
    ImageType::SizeType resolution = reader->GetOutput()->GetLargestPossibleRegion().GetSize();

    vResolution.resize(resolution.GetSizeDimension(),-1);
    for(size_t i=0;i<resolution.GetSizeDimension();i++)
      vResolution[i]=resolution[i];

    GateMessage("Image", 5, "GateDICOMImage::GetResolution: " << vResolution[0] <<","<< vResolution[1] <<","<< vResolution[2] << " mm" << Gateendl);
  }
  return vResolution;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
std::vector<double> GateDICOMImage::GetSpacing()
{
  if(vSpacing.size()==0)
  {
    ImageType::SpacingType spacing = reader->GetOutput()->GetSpacing();

    vSpacing.resize(reader->GetOutput()->GetImageDimension(),0.);
    for(size_t i=0;i<reader->GetOutput()->GetImageDimension();i++)
      vSpacing[i]=spacing[i];

    GateMessage("Image", 5, "GateDICOMImage::GetSpacing: " << vSpacing[0] <<","<< vSpacing[1] <<","<< vSpacing[2] << " mm" << Gateendl);
  }
  return vSpacing;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
std::vector<double> GateDICOMImage::GetImageSize()
{
  if(vSize.size()==0)
  {
    vSize.resize(reader->GetOutput()->GetImageDimension(),0.);
    for(size_t i=0;i<reader->GetOutput()->GetImageDimension();i++)
      vSize[i]=GetResolution()[i]*GetSpacing()[i];

    GateMessage("Image", 5, "GateDICOMImage::GetImageSize: " << vSize[0] <<","<< vSize[1] <<","<< vSize[2] << " mm" << Gateendl);
  }
  return vSize;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
std::vector<double> GateDICOMImage::GetOrigin()
{
  if(vOrigin.size()==0)
  {
    ImageType::PointType origin = reader->GetOutput()->GetOrigin();

    vOrigin.resize(reader->GetOutput()->GetImageDimension(),0.);
    for(size_t i=0;i<reader->GetOutput()->GetImageDimension();i++)
      vOrigin[i]=origin[i];

    GateMessage("Image", 5, "GateDICOMImage::GetOrigin: " << vOrigin[0] <<","<< vOrigin[1] <<","<< vOrigin[2] << Gateendl);
  }
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
