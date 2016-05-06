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
  pixelsCount=0;

  dicomIO=ImageType::New();

  GateMessage("Image", 5, "GateDICOMImage: dicomIO dimensions : " << dicomIO->GetImageDimension() << Gateendl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDICOMImage::Read(const std::string fileName)
{
  std::string path = gdcm::Filename(fileName.c_str()).GetPath();

  reader = ReaderType::New();
  reader->SetFileName(fileName);
  reader->SetImageIO(ImageIOType::New());

  try
  {
    reader->Update();
  }
  catch (itk::ExceptionObject & e)
  {
    GateError("Cannot read the file: " << fileName << Gateendl);
    exit(EXIT_FAILURE);
  }

  GateMessage("Image", 5, "GateDICOMImage::Read " << fileName << Gateendl);

  ReaderType::DictionaryRawPointer dict = (*(reader->GetMetaDataDictionaryArray()))[0];
  itk::MetaDataObject< std::string >::Pointer entryvalue =
    dynamic_cast<itk::MetaDataObject< std::string > *>( dict->Find( "0020|000e" )->second.GetPointer() );

  if( entryvalue )
  {
    std::string seriesUID(entryvalue->GetMetaDataObjectValue());
    GateMessage("Image", 5, "GateDICOMImage::Read: File: " << fileName <<", series UID: "<< seriesUID << Gateendl);
    ReadSeries(path,seriesUID);
  }
  else
  {
    GateError("Can't find the series UID from the file" << Gateendl);
    exit(EXIT_FAILURE);
  }
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

    GateMessage("Image", 5, "The directory " << seriesDirectory << " contains " << nameGenerator->GetSeriesUIDs().size() << " DICOM Series:" << Gateendl);

    std::vector< std::string >::const_iterator seriesItr = nameGenerator->GetSeriesUIDs().begin();
    std::vector< std::string >::const_iterator seriesEnd = nameGenerator->GetSeriesUIDs().end();

    while( seriesItr != seriesEnd )
    {
      std::string seriesUID(seriesItr->c_str());
      if(UID!="" && UID==seriesUID.substr(0, UID.size()))
        UID=seriesItr->c_str();
      //std::cout << "UID: " << UID << "seriesUID.substr: " << seriesUID.substr(0, UID.size()) << std::endl;
      GateMessage("Image", 5, "  " << seriesItr->c_str() << Gateendl);
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
    vResolution.resize(reader->GetOutput()->GetImageDimension(),-1);
    for(size_t i=0;i<reader->GetOutput()->GetImageDimension();i++)
      vResolution[i]=reader->GetOutput()->GetLargestPossibleRegion().GetSize()[i];

    GateMessage("Image", 5, "GateDICOMImage::GetResolution: " << vResolution[0] <<","<< vResolution[1] <<","<< vResolution[2] << Gateendl);
  }
  return vResolution;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
std::vector<double> GateDICOMImage::GetSpacing()
{
  if(vSpacing.size()==0)
  {
    vSpacing.resize(reader->GetOutput()->GetImageDimension(),0.);
    for(size_t i=0;i<reader->GetOutput()->GetImageDimension();i++)
      vSpacing[i]=reader->GetOutput()->GetSpacing()[i];

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
    vOrigin.resize(reader->GetOutput()->GetImageDimension(),0.);
    for(size_t i=0;i<reader->GetOutput()->GetImageDimension();i++)
      vOrigin[i]=reader->GetOutput()->GetOrigin()[i];

    GateMessage("Image", 5, "GateDICOMImage::GetOrigin: " << vOrigin[0] <<","<< vOrigin[1] <<","<< vOrigin[2] << Gateendl);
  }
  return vOrigin;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDICOMImage::SetResolution(std::vector<long unsigned int> resolution)
{
  //if(dicomIO->GetImageDimension()==3)
  //{
  //  GateError("GateDICOMImage::SetResolution: The image dimension is not 3 (it's " << dicomIO->GetImageDimension << ")" << Gateendl);
  //  exit(EXIT_FAILURE);
  //}

  ImageType::RegionType region;
  region.SetIndex({{0,0,0}});
  region.SetSize({{resolution[0],resolution[1],resolution[2]}});

  dicomIO->SetRegions(region);
  dicomIO->Allocate();

  GateMessage("Image", 5, "GateDICOMImage::SetResolution: "
            << dicomIO->GetLargestPossibleRegion().GetSize()[0] <<","
            << dicomIO->GetLargestPossibleRegion().GetSize()[1] <<","
            << dicomIO->GetLargestPossibleRegion().GetSize()[2] << Gateendl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDICOMImage::SetSpacing(std::vector<double> spacing)
{
  ImageType::SpacingType spacingST;

  for(size_t i=0;i<dicomIO->GetImageDimension();i++)
    spacingST[i]=spacing[i];

  dicomIO->SetSpacing(spacingST);

  GateMessage("Image", 5, "GateDICOMImage::SetSpacing: "
            << dicomIO->GetSpacing()[0] <<","
            << dicomIO->GetSpacing()[1] <<","
            << dicomIO->GetSpacing()[2] << Gateendl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDICOMImage::SetOrigin(std::vector<double> origin)
{
  ImageType::PointType originPT;

  for(size_t i=0;i<dicomIO->GetImageDimension();i++)
    originPT[i]=origin[i];

  dicomIO->SetOrigin(originPT);

  GateMessage("Image", 5, "GateDICOMImage::SetOrigin: "
            << dicomIO->GetOrigin()[0] <<","
            << dicomIO->GetOrigin()[1] <<","
            << dicomIO->GetOrigin()[2] << Gateendl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDICOMImage::dumpIO()
{
  GateMessage("Image", 5, "GateDICOMImage::dumpIO: dicomIO resolution : "
            << dicomIO->GetLargestPossibleRegion().GetSize()[0] <<","
            << dicomIO->GetLargestPossibleRegion().GetSize()[1] <<","
            << dicomIO->GetLargestPossibleRegion().GetSize()[2] << Gateendl);

  GateMessage("Image", 5, "GateDICOMImage::dumpIO: dicomIO spacing : "
            << dicomIO->GetSpacing()[0] <<","
            << dicomIO->GetSpacing()[1] <<","
            << dicomIO->GetSpacing()[2] << Gateendl);

  GateMessage("Image", 5, "GateDICOMImage::dumpIO: dicomIO origin : "
            << dicomIO->GetOrigin()[0] <<","
            << dicomIO->GetOrigin()[1] <<","
            << dicomIO->GetOrigin()[2] << Gateendl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDICOMImage::Write(const std::string fileName)
{
  if(fileName=="")
    std::cerr << "ERROR: No filename given for the exported image !" << std::endl;

  itk::ImageFileWriter<ImageType>::Pointer writer = itk::ImageFileWriter<ImageType>::New();

  writer->SetFileName(fileName);
  writer->SetInput(dicomIO);
  dumpIO();

  GateMessage("Image", 5, "GateDICOMImage::Write: Writing the image as " << fileName << Gateendl);

  try
  {
    writer->Update();
  }
  catch (itk::ExceptionObject &excp)
  {
    std::cerr << "Exception thrown while writing the series" << std::endl;
    std::cerr << excp << std::endl;
    exit(EXIT_FAILURE);
  }

}
//-----------------------------------------------------------------------------
