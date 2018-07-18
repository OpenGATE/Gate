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

#include "GateDICOMImage.hh"

//#if ITK_VERSION_MAJOR >= 4
//#if ( ( ITK_VERSION_MAJOR == 4 ) && ( ITK_VERSION_MINOR < 6 ) )

//-----------------------------------------------------------------------------
GateDICOMImage::GateDICOMImage()
{
  vResolution.resize(0);
  vSpacing.resize(0);
  vSize.resize(0);
  vOrigin.resize(0);
  pixelsCount=0;

  dicomIO=ImageType::New();

  // This GateMessage changes the cout precision to the number of decimal digits necessary to differentiate all values of double type.
  GateMessage("Image", 15, std::setprecision(std::numeric_limits<double>::digits10) << Gateendl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDICOMImage::Read(const std::string fileName)
{
  std::string path = gdcm::Filename(fileName.c_str()).GetPath();

  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(fileName);
  reader->SetImageIO(ImageIOType::New());

  try
  {
    reader->Update();
  }
  catch (itk::ExceptionObject & e)
  {
    GateError("[GateDICOMImage::" << __FUNCTION__ << "] ERROR: Cannot read the file " << fileName << Gateendl);
    exit(EXIT_FAILURE);
  }

  GateMessage("Image", 5, "[GateDICOMImage::" << __FUNCTION__ << "] Opening " << fileName << Gateendl);

  ReaderType::DictionaryRawPointer dict = (*(reader->GetMetaDataDictionaryArray()))[0];
  itk::MetaDataObject< std::string >::Pointer entryvalue =
    dynamic_cast<itk::MetaDataObject< std::string > *>( dict->Find( "0020|000e" )->second.GetPointer() );

  if( entryvalue )
  {
    std::string seriesUID(entryvalue->GetMetaDataObjectValue());
    GateMessage("Image", 5, "[GateDICOMImage::" << __FUNCTION__ << "] File: " << fileName <<", series UID: "<< seriesUID << Gateendl);
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
  GateMessage("Image", 5, "[GateDICOMImage::" << __FUNCTION__ << "]: path: " << seriesDirectory <<", UID: "<< UID << Gateendl);

  ReaderType::Pointer reader = ReaderType::New();
  reader->SetImageIO(ImageIOType::New());

  NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();
  nameGenerator->SetUseSeriesDetails( true );
  nameGenerator->SetDirectory( seriesDirectory );

  if(nameGenerator->GetSeriesUIDs().size() == 0)
  {
    GateError( "[GateDICOMImage::" << __FUNCTION__ << "] ERROR:" << seriesDirectory << " does not contain any DICOM series !" << Gateendl);
    exit(EXIT_FAILURE);
  }

  std::vector< std::string >::const_iterator seriesItr = nameGenerator->GetSeriesUIDs().begin();
  std::vector< std::string >::const_iterator seriesEnd = nameGenerator->GetSeriesUIDs().end();

  bool found(false);
  while( seriesItr != seriesEnd )
  {
    std::string seriesUID(seriesItr->c_str());
    if(UID != "" && UID == seriesUID.substr(0, UID.size()))
    {
      UID=seriesItr->c_str();
      GateMessage("Image", 2, "[GateDICOMImage::" << __FUNCTION__ << "] Corresponding DICOM series found ! (UID: " << seriesItr->c_str() << ")" << Gateendl);
      found = true;
    }
    ++seriesItr;
  }

  if(!found)
  {
    GateError( "[GateDICOMImage::" << __FUNCTION__ << "] ERROR:" << seriesDirectory << " does not contain any corresponding DICOM series !" << Gateendl);
    exit(EXIT_FAILURE);
  }

  if(UID != "")
    reader->SetFileNames(nameGenerator->GetFileNames(UID));

  try
  {
    reader->Update();
  }
  catch (itk::ExceptionObject &excp)
  {
    GateError( "[GateDICOMImage::" << __FUNCTION__ << "] ERROR:" << seriesDirectory << " does not contain any corresponding DICOM series !" << Gateendl);
    exit(EXIT_FAILURE);
  }

  image = reader->GetOutput();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
std::vector<long int unsigned> GateDICOMImage::GetResolution()
{
  if(vResolution.size() == 0)
  {
    vResolution.resize(image->GetImageDimension(), -1);
    for(size_t i = 0; i < image->GetImageDimension(); i++)
      vResolution[i] = image->GetLargestPossibleRegion().GetSize()[i];

    GateMessage("Image", 5, "[GateDICOMImage::" << __FUNCTION__ << "] "
                << vResolution[0] << ","
                << vResolution[1] << ","
                << vResolution[2] << Gateendl);
  }
  return vResolution;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
std::vector<double> GateDICOMImage::GetSpacing()
{
  if(vSpacing.size() == 0)
  {
    vSpacing.resize(image->GetImageDimension(), 0.);
    for(size_t i = 0; i < image->GetImageDimension(); i++)
      // NOTE: Rounding the spacing at six digits is mandatory to have the same precision for the DCM images and for the MHD images (the precision of the spacing in MHD images is rounded at six digits).
      vSpacing[i] = round_to_digits(image->GetSpacing()[i],6);

    GateMessage("Image", 5, "[GateDICOMImage::" << __FUNCTION__ << "] "
                << vSpacing[0] << ","
                << vSpacing[1] << ","
                << vSpacing[2] << " mm" << Gateendl);
  }
  return vSpacing;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
std::vector<double> GateDICOMImage::GetSize()
{
  if(vSize.size() == 0)
  {
    vSize.resize(image->GetImageDimension(), 0);
    for(size_t i=0 ; i<image->GetImageDimension() ; i++)
      vSize[i] = GetResolution()[i] * GetSpacing()[i];

    GateMessage("Image", 5, "[GateDICOMImage::" << __FUNCTION__ << "] "
                << vSize[0] << ","
                << vSize[1] << ","
                << vSize[2] << " mm" << Gateendl);
  }
  return vSize;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
std::vector<double> GateDICOMImage::GetOrigin()
{
  if(vOrigin.size() == 0)
  {
    vOrigin.resize(image->GetImageDimension(),0.);
    for(size_t i=0 ; i<image->GetImageDimension() ; i++)
      // NOTE: Rounding the origin at six digits is mandatory to have the same precision for the DCM images and for the MHD images (the precision of the origin in MHD images is rounded at six digits).
      vOrigin[i] = round_to_digits(image->GetOrigin()[i],6);

    GateMessage("Image", 5, "[GateDICOMImage::" << __FUNCTION__ << "] "
                << vOrigin[0] <<","
                << vOrigin[1] <<","
                << vOrigin[2] << Gateendl);
  }
  return vOrigin;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4RotationMatrix GateDICOMImage::GetRotationMatrix()
{
  rotationMatrix.set(G4ThreeVector(image->GetDirection().GetVnlMatrix().get(0,0),image->GetDirection().GetVnlMatrix().get(0,1),image->GetDirection().GetVnlMatrix().get(0,2)),
                     G4ThreeVector(image->GetDirection().GetVnlMatrix().get(1,0),image->GetDirection().GetVnlMatrix().get(1,1),image->GetDirection().GetVnlMatrix().get(1,2)),
                     G4ThreeVector(image->GetDirection().GetVnlMatrix().get(2,0),image->GetDirection().GetVnlMatrix().get(2,1),image->GetDirection().GetVnlMatrix().get(2,2)));

  return rotationMatrix;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDICOMImage::SetResolution(std::vector<long unsigned int> resolution)
{
  ImageType::RegionType region;
  region.SetIndex({{0,0,0}});
  region.SetSize({{resolution[0],resolution[1],resolution[2]}});

  dicomIO->SetRegions(region);
  dicomIO->Allocate();

  GateMessage("Image", 5, "[GateDICOMImage::" << __FUNCTION__ << "] "
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

  GateMessage("Image", 5, "[GateDICOMImage::" << __FUNCTION__ << "] "
            << dicomIO->GetSpacing()[0] <<","
            << dicomIO->GetSpacing()[1] <<","
            << dicomIO->GetSpacing()[2] << Gateendl);

  std::ostringstream value;
  value.str("");
  value << dicomIO->GetSpacing()[2];
  itk::MetaDataDictionary & mainDictonary = (dicomIO->GetMetaDataDictionary());
  itk::EncapsulateMetaData<std::string> (mainDictonary, "0018|0050", value.str());//SpacingBetweenSlices
  itk::EncapsulateMetaData<std::string> (mainDictonary, "0018|0088", value.str());//SliceThickness
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDICOMImage::SetOrigin(std::vector<double> origin)
{
  // Gate convention: origin is the corner of the first pixel
  // MHD / ITK convention: origin is the center of the first pixel
  // -> Add a half pixel
  GateMessage("Image", 5, "[GateDICOMImage::" << __FUNCTION__ << "] Untouched origin: "
                << origin[0] << ","
                << origin[1] << ","
                << origin[2] << Gateendl);

  //p[0] = image->GetOrigin().x() + image->GetVoxelSize().x()/2.0;

  ImageType::PointType originPT;

  for(size_t i=0;i<dicomIO->GetImageDimension();i++)
    originPT[i] = origin[i] + dicomIO->GetSpacing()[i]/2.0;

  dicomIO->SetOrigin(originPT);

  GateMessage("Image", 5, "[GateDICOMImage::" << __FUNCTION__ << "] Conventional origin: "
            << dicomIO->GetOrigin()[0] << ","
            << dicomIO->GetOrigin()[1] << ","
            << dicomIO->GetOrigin()[2] << Gateendl);

  std::ostringstream valueX,valueY,valueZ;
  valueX.str("");
  valueY.str("");
  valueZ.str("");
  valueX << dicomIO->GetOrigin()[0];
  valueY << dicomIO->GetOrigin()[1];
  valueZ << dicomIO->GetOrigin()[2];
  itk::MetaDataDictionary & mainDictonary = (dicomIO->GetMetaDataDictionary());
  itk::EncapsulateMetaData<std::string> (mainDictonary, "0020|0032", valueX.str()+"\\"+valueY.str()+"\\"+valueZ.str());//Image Position (Patient)
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDICOMImage::dumpIO()
{
  GateMessage("Image", 5, "[GateDICOMImage::" << __FUNCTION__ << "] dicomIO resolution : "
            << dicomIO->GetLargestPossibleRegion().GetSize()[0] <<","
            << dicomIO->GetLargestPossibleRegion().GetSize()[1] <<","
            << dicomIO->GetLargestPossibleRegion().GetSize()[2] << Gateendl);

  GateMessage("Image", 5, "[GateDICOMImage::" << __FUNCTION__ << "] dicomIO spacing : "
            << dicomIO->GetSpacing()[0] <<","
            << dicomIO->GetSpacing()[1] <<","
            << dicomIO->GetSpacing()[2] << Gateendl);

  GateMessage("Image", 5, "[GateDICOMImage::" << __FUNCTION__ << "] dicomIO origin : "
            << dicomIO->GetOrigin()[0] <<","
            << dicomIO->GetOrigin()[1] <<","
            << dicomIO->GetOrigin()[2] << Gateendl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDICOMImage::Write(const std::string fileName)
{
  // EXPERIMENTAL METHOD !! FIXME
  GateMessage("Image", 0, "[GateDICOMImage::" << __FUNCTION__ << "] WARNING: DICOM writing is experimental !" << fileName << Gateendl);

  if(fileName == "")
  {
    GateError("[GateDICOMImage::" << __FUNCTION__ << "] ERROR: No filename given for the exported image !" << fileName << Gateendl);
    exit(EXIT_FAILURE);
  }

  itk::ImageFileWriter<ImageType>::Pointer writer (itk::ImageFileWriter<ImageType>::New());

  writer->SetFileName(fileName);
  writer->SetInput(dicomIO);
  writer->UseInputMetaDataDictionaryOn(); //MANDATORY
  writer->SetImageIO(ImageIOType::New());
  //dumpIO();

  GateMessage("Image", 5, "[GateDICOMImage::" << __FUNCTION__ << "] Writing the image as " << fileName << Gateendl);

  try {writer->Update();}
  catch (itk::ExceptionObject &excp)
  {
    GateError("[GateDICOMImage::" << __FUNCTION__ << "] ERROR: Exception thrown while writing " << fileName << Gateendl);
    exit(EXIT_FAILURE);
  }
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
double GateDICOMImage::round_to_digits(double value, int digits)
{
  if (value == 0.0)
    return 0.0;

  double factor (pow(10.0, digits - ceil(log10(fabs(value)))));
  return round(value * factor) / factor;
}
//-----------------------------------------------------------------------------

#endif
