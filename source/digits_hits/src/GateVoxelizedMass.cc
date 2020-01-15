/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*
  \class  GateVoxelizedMass
  \author Thomas DESCHLER (thomas.deschler@iphc.cnrs.fr)
  \date	October 2015
*/

/*
  \brief Class GateVoxelizedMass :
  \brief
*/

#include "GateVoxelizedMass.hh"
#include "GateMiscFunctions.hh"
#include "GateDetectorConstruction.hh"
#include "GateMaterialDatabase.hh"

#include <G4IntersectionSolid.hh>
#include <G4SubtractionSolid.hh>
#include <G4PhysicalVolumeStore.hh>
#include <G4VSolid.hh>
#include <G4Box.hh>
#include <G4VPVParameterisation.hh>

#include <ctime>
#include <cstring>


//-----------------------------------------------------------------------------
GateVoxelizedMass::GateVoxelizedMass()
{
  mIsInitialized        = false;
  mIsParameterised      = false;
  mIsVecGenerated       = false;
  mHasFilter            = false;
  mHasExternalMassImage = false;
  mHasSameResolution    = false;

  mMassFile       = "";
  mMaterialFilter = "";

  mCubicVolume.clear();
  mMass       .clear();
  mEdep       .clear();

  doselReconstructedMass.clear();
  doselExternalMass     .clear();

  imageVolume = 0;
  imageVoxel  = 0;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVoxelizedMass::Initialize(const G4String _mVolumeName, const GateVImage* _mImage)
{
  GateMessage("Actor", 10, "[GateVoxelizedMass::" << __FUNCTION__ << "] Started" << Gateendl);

  mIsVecGenerated    = false;
  mHasSameResolution = false;

  mVolumeName = _mVolumeName;
  mImage      = _mImage;

  DAPV = G4PhysicalVolumeStore::GetInstance()->GetVolume(mVolumeName+"_phys");
  DALV = DAPV->GetLogicalVolume();

  mIsParameterised = IsLVParameterized(DALV);

  GateMessage("Actor", 1, "[GateVoxelizedMass::" << __FUNCTION__ << "] Is parameterised ? " <<  mIsParameterised << Gateendl);

  if (!mIsParameterised) {
    mCubicVolume.resize(mImage->GetNumberOfValues());
    mMass       .resize(mImage->GetNumberOfValues());
    mEdep       .resize(mImage->GetNumberOfValues());

    for(signed long int index=0; index < mImage->GetNumberOfValues(); index++) {
      mCubicVolume[index].clear();
      mMass       [index].clear();
      mEdep       [index].clear();
    }

    doselSV=new G4Box("DoselSV",
                      mImage->GetVoxelSize().getX()/2.0,
                      mImage->GetVoxelSize().getY()/2.0,
                      mImage->GetVoxelSize().getZ()/2.0);
  }

  doselExternalMass.clear();
  if (mHasExternalMassImage && mMassFile != "")
    {
      mMassImage = new GateImageDouble();
      mMassImage->Read(mMassFile);

      if (mMassImage->GetResolution() != mImage->GetResolution())
        GateError("Error: " << mMassFile << " hasn't the right resolution !" << Gateendl <<
                  "   Actor resolution: " << mImage->GetResolution() << Gateendl <<
                  "   " << mMassFile << " resolution: " << mMassImage->GetResolution() << Gateendl);

      if (mMassImage->GetNumberOfValues() != mImage->GetNumberOfValues())
        GateError("Error: " << mMassFile << " hasn't the right number of dosels !" << Gateendl <<
                  "   Actor number of dosels: " << mImage->GetNumberOfValues() << Gateendl <<
                  "   " << mMassFile << " number of dosels: " << mMassImage->GetNumberOfValues() << Gateendl);

      double diff((mMassImage->GetVoxelVolume()-mImage->GetVoxelVolume())*100/mImage->GetVoxelVolume());
      double substractionError(0.5);

      if (std::abs(diff) > substractionError)
        GateError("Error: " << mMassFile << " hasn't the right dosel cubic volume !" << Gateendl <<
                  "   Actor dosel cubic volume: " << G4BestUnit(mImage->GetVoxelVolume(),"Volume") << Gateendl <<
                  "   " << mMassFile << " dosel cubic volume: " << G4BestUnit(mMassImage->GetVoxelVolume(),"Volume") << Gateendl <<
                  "   Difference: " << diff << Gateendl);

      doselExternalMass.resize(mMassImage->GetNumberOfValues(),-1.);

      for(signed long int i=0; i < mMassImage->GetNumberOfValues(); i++)
        doselExternalMass[i]=mMassImage->GetValue(i)*kg;
    }

  //vectorSV.clear();


  if (mIsParameterised) {
    imageVolume = dynamic_cast<GateVImageVolume*>(GateObjectStore::GetInstance()->FindVolumeCreator(mVolumeName));
    imageVoxel  = imageVolume->GetImage();

    if (doselExternalMass.size() == 0)
      GenerateVoxels();
  }

  GateMessage("Actor", 1,  "[GateVoxelizedMass::" << __FUNCTION__ << "] Has same resolution ? " << mHasSameResolution << Gateendl);

  if (!mHasSameResolution) {
    doselReconstructedMass.clear();
    doselReconstructedMass.resize(mImage->GetNumberOfValues(),-1.);
  }

  if (mHasFilter) {
    if (mIsParameterised && mVolumeFilter != "") {
      GateError( "[GateVoxelizedMass::" << __FUNCTION__ << "] ERROR: volume filter is not compatible with parameterised volumes !" << Gateendl);
      exit(EXIT_FAILURE);
    }
    else if (!mIsParameterised && mMaterialFilter != "") {
      GateError( "[GateVoxelizedMass::" << __FUNCTION__ << "] ERROR: material filter is only compatible with parameterised volumes !" << Gateendl);
      exit(EXIT_FAILURE);
    }
  }

  mIsInitialized=true;

  GateMessage("Actor", 10, "[GateVoxelizedMass::" << __FUNCTION__ << "] Ended" << Gateendl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool GateVoxelizedMass::IsLVParameterized(const G4LogicalVolume* LV)
{
  if (LV->GetSolid()->GetEntityType() == "GateImageRegionalizedVolumeSolid")
    return true;

  if (LV->GetNoDaughters()==1)
    if (LV->GetDaughter(0)->IsParameterised() ||
        (LV->GetDaughter(0)->GetName().find("voxel_phys_Y")!=std::string::npos &&
         LV->GetDaughter(0)->GetLogicalVolume()->GetDaughter(0)->GetName().find("voxel_phys_X")!=std::string::npos &&
         LV->GetDaughter(0)->GetLogicalVolume()->GetDaughter(0)->GetLogicalVolume()->GetDaughter(0)->GetName().find("voxel_phys_Z")!=std::string::npos))
      return true;

  return false;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateVoxelizedMass::GetDoselMass(const int index)
{
  GateMessage("Actor", 10, "[GateVoxelizedMass::" << __FUNCTION__ << "] Started" << Gateendl);

  if (mHasSameResolution) {
    GateMessage("Actor", 10,  "[GateVoxelizedMass::" << __FUNCTION__ << "] Volume and actor resolution are the same ! I will simply read the mass of the voxel." << Gateendl);

    return theMaterialDatabase.GetMaterial((G4String)imageVolume->GetMaterialNameFromLabel(imageVolume->GetImage()->GetValue(index)))->GetDensity() * imageVolume->GetImage()->GetVoxelVolume();
  }

  // Case of imported mass image
  if (mHasExternalMassImage)
    {
      if (doselExternalMass.size() > 0.)
        return doselExternalMass[index];
      else
        GateError( "[GateVoxelizedMass::" << __FUNCTION__ << "] ERROR: initialization of doselExternalMass is incorrect !" << Gateendl);
    }

  if (doselReconstructedMass.size()        == 0) {
    doselReconstructedMass.clear();
    doselReconstructedMass.resize(mImage->GetNumberOfValues(),-1.);
  }

  if(doselReconstructedMass[index] < 0.)
    {
      GateMessage("Actor", 11,  "[GateVoxelizedMass::" << __FUNCTION__ << "] I don't have the mass of this voxel (index: " << index << ")" << Gateendl);

      if (mIsParameterised)
        doselReconstructedData = ParameterizedVolume(index);
      else
        doselReconstructedData = VoxelIteration(DAPV,
                                                0,
                                                DAPV->GetObjectRotationValue(),
                                                DAPV->GetObjectTranslation(),
                                                index);

      doselReconstructedMass[index]        = doselReconstructedData.first;
    }

  GateMessage("Actor", 11,  "[GateVoxelizedMass::" << __FUNCTION__ << "] Computed mass of voxel (index: " << index << "):" << doselReconstructedMass[index] << Gateendl);

  GateMessage("Actor", 10, "[GateVoxelizedMass::" << __FUNCTION__ << "] Ended" << Gateendl);

  return doselReconstructedMass[index];
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateVoxelizedMass::GetDoselVolume()
{
  if (mHasSameResolution)
    return imageVoxel->GetVoxelVolume();

  return mImage->GetVoxelVolume();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
std::vector<double> GateVoxelizedMass::GetDoselMassVector()
{
  if (doselExternalMass.size() > 0)
    return doselExternalMass;

  if (!mIsVecGenerated)
    GenerateVectors();

  return doselReconstructedMass;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVoxelizedMass::GenerateVectors()
{
  GateMessage("Actor", 10, "[GateVoxelizedMass::" << __FUNCTION__ << "] Started" << Gateendl);

  GateMessage("Actor", 0,  "[GateVoxelizedMass::" << __FUNCTION__ << "] Total voxelized mass calculation in progress, please wait ... " << Gateendl);

  GateMessage("Actor", 1, "[GateVoxelizedMass::" << __FUNCTION__ << "] Number of values in the images: " <<  mImage->GetNumberOfValues() << Gateendl);

  time_t timer1,timer2,timer3,timer4;
  time(&timer1);

  if (doselReconstructedMass.size() == 0)
    {
      doselReconstructedMass.clear();
      doselReconstructedMass.resize(mImage->GetNumberOfValues(),-1.);
    }

  doselReconstructedTotalCubicVolume = 0.;
  doselReconstructedTotalMass        = 0.;

  for(signed long int i=0; i < mImage->GetNumberOfValues(); i++)
    {
      time(&timer3);

      if(mIsParameterised)
        doselReconstructedData = ParameterizedVolume(i);
      else
        doselReconstructedData = VoxelIteration(DAPV,
                                                0,
                                                DAPV->GetObjectRotationValue(),
                                                DAPV->GetObjectTranslation(),
                                                i);

      doselReconstructedMass[i]        = doselReconstructedData.first;

      doselReconstructedTotalMass        += doselReconstructedMass[i];
      doselReconstructedTotalCubicVolume += doselReconstructedData.second;

      time(&timer4);
      seconds=difftime(timer4,timer1);

      if (difftime(timer4,timer1) >= 60 && i%100 == 0)
        {
          std::cout<<" "<<i*100/mImage->GetNumberOfValues()<<"% (time elapsed : "<<seconds/60<<"min"<<seconds%60<<"s)      \r"<<std::flush;
          // Experimental
          /*seconds=(mImage->GetNumberOfValues()-i)*difftime(timer4,timer3);
            if(seconds!=0.) std::cout<<"Estimated remaining time : "<<seconds/60<<"min"<<seconds%60<<"s ("<<seconds<<"s)                \r"<<std::flush;*/
        }
    }

  time(&timer2);
  seconds=difftime(timer2,timer1);

  G4String s="\n";
  if (mIsParameterised)
    s = "\tNumber of voxels : "+DoubletoString(imageVoxel->GetNumberOfValues())+'\n';

  GateMessage("Actor", 1,
              "[GateVoxelizedMass] Summary: mass calculation for voxelized volume :" << G4endl
              <<"\tTime elapsed : " << seconds/60 << "min" << seconds%60 << "s (" << seconds << "s)" << G4endl
              << s
              << "\tNumber of dosels : " << mImage->GetNumberOfValues() << G4endl
              << "\tDosels reconstructed total mass : " << G4BestUnit(doselReconstructedTotalMass,"Mass") << G4endl
              << "\tDosels reconstructed total cubic volume : "
              << G4BestUnit(doselReconstructedTotalCubicVolume,"Volume") << G4endl);

  mIsVecGenerated=true;

  GateMessage("Actor", 10, "[GateVoxelizedMass::" << __FUNCTION__ << "] Ended" << Gateendl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVoxelizedMass::GenerateVoxels()
{
  unsigned long int nxVoxel = imageVoxel->GetResolution().x();
  unsigned long int nyVoxel = imageVoxel->GetResolution().y();
  unsigned long int nzVoxel = imageVoxel->GetResolution().z();

  DABox = (G4Box*)DALV->GetSolid();

  const unsigned long int nxDosel = round(DABox->GetXHalfLength()/(mImage->GetVoxelSize().x()/2.));
  const unsigned long int nyDosel = round(DABox->GetYHalfLength()/(mImage->GetVoxelSize().y()/2.));
  const unsigned long int nzDosel = round(DABox->GetZHalfLength()/(mImage->GetVoxelSize().z()/2.));


  GateMessage("Actor", 5, "[GateVoxelizedMass::" << __FUNCTION__ << "] Voxel resolution: " << nxVoxel << "," << nyVoxel << "," << nzVoxel << Gateendl);
  GateMessage("Actor", 5, "[GateVoxelizedMass::" << __FUNCTION__ << "] Dosel resolution: " << nxDosel << "," << nyDosel << "," << nzDosel << Gateendl);
  GateMessage("Actor", 5, "[GateVoxelizedMass::" << __FUNCTION__ << "] Total voxels    : " << imageVoxel->GetNumberOfValues() << Gateendl);
  GateMessage("Actor", 5, "[GateVoxelizedMass::" << __FUNCTION__ << "] Total dosels    : " << nxDosel * nyDosel * nzDosel << Gateendl);


  if ((nxDosel + 2 == nxVoxel &&
       nyDosel + 2 == nyVoxel &&
       nzDosel + 2 == nzVoxel) ||
      DALV->GetSolid()->GetEntityType() == "GateImageRegionalizedVolumeSolid")
    {
      GateMessage("Actor", 2, "[GateVoxelizedMass::" << __FUNCTION__ << "] Voxel margins detected in image ! Generating a new image with no margins !" << Gateendl);

      GateImage* newImage = new GateImage;

      G4ThreeVector res(imageVoxel->GetResolution().x() - 2,
                        imageVoxel->GetResolution().y() - 2,
                        imageVoxel->GetResolution().z() - 2);
      newImage->SetResolutionAndVoxelSize(res,imageVoxel->GetVoxelSize());
      newImage->SetOrigin(imageVoxel->GetOrigin());
      newImage->SetTransformMatrix(imageVoxel->GetTransformMatrix());
      newImage->Allocate();
      newImage->SetOutsideValue(imageVoxel->GetMinValue() - 1);
      newImage->Fill(newImage->GetOutsideValue());

      unsigned long int i,j,k;
      for (k=1;k<imageVoxel->GetResolution().z()-1;k++)
        for (j=1;j<imageVoxel->GetResolution().y()-1;j++)
          for (i=1;i<imageVoxel->GetResolution().x()-1;i++)
            newImage->SetValue(i-1,j-1,k-1,imageVoxel->GetValue(i,j,k));

      imageVoxel = newImage;

      nxVoxel = imageVoxel->GetResolution().x();
      nyVoxel = imageVoxel->GetResolution().y();
      nzVoxel = imageVoxel->GetResolution().z();

      GateMessage("Actor", 5, "[GateVoxelizedMass::" << __FUNCTION__ << "] New total voxels: " << imageVoxel->GetNumberOfValues() << Gateendl);
    }


  if ( nxDosel == nxVoxel &&
       nyDosel == nyVoxel &&
       nzDosel == nzVoxel ) {
    mHasSameResolution = true;

    GateMessage("Actor", 2, "[GateVoxelizedMass::" << __FUNCTION__ << "] Resolution of voxels and dosels are the same !" << Gateendl);

    return;
  }


  if (nxDosel > nxVoxel ||
      nyDosel > nyVoxel ||
      nzDosel > nzVoxel)
    {
      GateError("[GateVoxelizedMass::" << __FUNCTION__ << "] ERROR : The dosel resolution is smaller than the voxel resolution !" << Gateendl);
      exit(EXIT_FAILURE);
    }


  for (signed long int i=0; i < imageVoxel->GetNumberOfValues(); i++)
    {
      const unsigned long int xVoxel(round((DABox->GetXHalfLength()+imageVoxel->GetVoxelCenterFromIndex(i).x()-imageVoxel->GetVoxelSize().x()/2.)/imageVoxel->GetVoxelSize().x()));
      const unsigned long int yVoxel(round((DABox->GetYHalfLength()+imageVoxel->GetVoxelCenterFromIndex(i).y()-imageVoxel->GetVoxelSize().y()/2.)/imageVoxel->GetVoxelSize().y()));
      const unsigned long int zVoxel(round((DABox->GetZHalfLength()+imageVoxel->GetVoxelCenterFromIndex(i).z()-imageVoxel->GetVoxelSize().z()/2.)/imageVoxel->GetVoxelSize().z()));

      //G4cout<<G4endl;
      //G4cout<<"Index="<<i<<G4endl;
      //G4cout<<"xVoxel="<<xVoxel<<G4endl;
      //G4cout<<"yVoxel="<<yVoxel<<G4endl;
      //G4cout<<"zVoxel="<<zVoxel<<G4endl;

      if (imageVoxel->GetValue(i) != imageVoxel->GetValue(xVoxel,yVoxel,zVoxel))
        {
          GateError("[GateVoxelizedMass::" << __FUNCTION__ << "] ERROR: Reconstructed coordinates of voxels don't correspond to index ! (index: " << i << ", coord: " << xVoxel << "," << yVoxel << "," << zVoxel << ")" << Gateendl);
          exit(EXIT_FAILURE);
        }

      if (xVoxel >= nxVoxel ||
          yVoxel >= nyVoxel ||
          zVoxel >= nzVoxel)
        {
          GateError("[GateVoxelizedMass::" << __FUNCTION__ << "] ERROR: Too many voxels ! (xVoxel = " << xVoxel << ", yVoxel = " << yVoxel << ", zVoxel = " << zVoxel << ")" << Gateendl);
          exit(EXIT_FAILURE);
        }
    }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4String GateVoxelizedMass::GetVoxelMatName(const unsigned long int x, const unsigned long int y, const unsigned long int z)
{
  GateMessage("Actor", 10, "[GateVoxelizedMass::" << __FUNCTION__ << "] Started" << Gateendl);

  const G4String voxelMatName = imageVolume->GetMaterialNameFromLabel(imageVolume->GetImage()->GetValue(x,y,z));

  GateMessage("Actor", 10, "[GateVoxelizedMass::" << __FUNCTION__ << "] Ended" << Gateendl);

  return voxelMatName;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4double GateVoxelizedMass::GetVoxelVolume()
{
  GateMessage("Actor", 10, "[GateVoxelizedMass::" << __FUNCTION__ << "] Started" << Gateendl);

  const G4double volume = imageVolume->GetImage()->GetVoxelVolume();

  if (volume <= 0.)
    GateError("[GateVoxelizedMass::" << __FUNCTION__ << "] ERROR: Voxel cubic volume is less or equal to zero ! (volume: " << volume << ")" << Gateendl);

  GateMessage("Actor", 10, "[GateVoxelizedMass::" << __FUNCTION__ << "] Ended" << Gateendl);

  return volume;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
G4double GateVoxelizedMass::GetVoxelMass(const unsigned long int x, const unsigned long int y, const unsigned long int z)
{
  GateMessage("Actor", 10, "[GateVoxelizedMass::" << __FUNCTION__ << "] Started" << Gateendl);

  const G4double density = theMaterialDatabase.GetMaterial(GetVoxelMatName(x,y,z))->GetDensity();
  const G4double mass    = density * GetVoxelVolume();

  if (mass <= 0.)
    {
      GateError("[GateVoxelizedMass::" << __FUNCTION__ << "] ERROR: Voxel (coord: " << x << "," << y << "," << z <<") mass is less or equal to zero ! (mass: "<< mass << ")" << Gateendl);
      exit(EXIT_FAILURE);
    }

  GateMessage("Actor", 10, "[GateVoxelizedMass::" << __FUNCTION__ << "] Ended" << Gateendl);

  return mass;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVoxelizedMass::GenerateDosels(const int index)
{
  GateMessage("Actor", 10, "[GateVoxelizedMass::" << __FUNCTION__ << "] Started" << Gateendl);
  // INFO : Dimension of the vectors : x = 0, y = 1, z = 2

  doselMin.resize(3,-1.);
  doselMin[0]=(DABox->GetXHalfLength()+mImage->GetVoxelCenterFromIndex(index).getX()-mImage->GetVoxelSize().getX()/2.)/imageVolume->GetImage()->GetVoxelSize().x();
  doselMin[1]=(DABox->GetYHalfLength()+mImage->GetVoxelCenterFromIndex(index).getY()-mImage->GetVoxelSize().getY()/2.)/imageVolume->GetImage()->GetVoxelSize().y();
  doselMin[2]=(DABox->GetZHalfLength()+mImage->GetVoxelCenterFromIndex(index).getZ()-mImage->GetVoxelSize().getZ()/2.)/imageVolume->GetImage()->GetVoxelSize().z();

  const double sensibility (1e8);

  if (doselMin[0] < 0 && doselMin[0] > -sensibility)
    doselMin[0] = 0.;
  else if (doselMin[0] < 0)
    {
      GateError("[GateVoxelizedMass::" << __FUNCTION__ << "] ERROR : doselMin[0] < 0 ! (doselMin[0]: " << doselMin[0] << ", index: " << index << ")" << Gateendl);
      exit(EXIT_FAILURE);
    }

  if (doselMin[1] < 0 && doselMin[1] > -sensibility)
    doselMin[1] = 0.;
  else if (doselMin[1] < 0)
    {
      GateError("[GateVoxelizedMass::" << __FUNCTION__ << "] ERROR : doselMin[1] < 0 ! (doselMin[1]: " << doselMin[1] << ", index: " << index << ")" << Gateendl);
      exit(EXIT_FAILURE);
    }

  if (doselMin[2] < 0 && doselMin[2] > -sensibility)
    doselMin[2] = 0.;
  else if (doselMin[2] < 0)
    {
      GateError("[GateVoxelizedMass::" << __FUNCTION__ << "] ERROR : doselMin[2] < 0 ! (doselMin[2]: " << doselMin[2] << ", index: " << index << ")" << Gateendl);
      exit(EXIT_FAILURE);
    }

  doselMax.resize(3,-1.);
  doselMax[0]=(DABox->GetXHalfLength()+mImage->GetVoxelCenterFromIndex(index).getX()+mImage->GetVoxelSize().getX()/2.)/imageVolume->GetImage()->GetVoxelSize().x();
  doselMax[1]=(DABox->GetYHalfLength()+mImage->GetVoxelCenterFromIndex(index).getY()+mImage->GetVoxelSize().getY()/2.)/imageVolume->GetImage()->GetVoxelSize().y();
  doselMax[2]=(DABox->GetZHalfLength()+mImage->GetVoxelCenterFromIndex(index).getZ()+mImage->GetVoxelSize().getZ()/2.)/imageVolume->GetImage()->GetVoxelSize().z();

  GateMessage("Actor", 10, "[GateVoxelizedMass::" << __FUNCTION__ << "] Ended" << Gateendl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
pair<double,double> GateVoxelizedMass::ParameterizedVolume(const int index)
{
  GateMessage("Actor", 10, "[GateVoxelizedMass::" << __FUNCTION__ << "] Started" << Gateendl);

  GenerateDosels(index);

  G4double doselReconstructedVolume = 0.;
  doselReconstructedMass[index]     = 0.;

  //GateMessage("Actor", 10, "[GateVoxelizedMass::" << __FUNCTION__ << "] DEBUG: doselMin[0]: " << doselMin[0] << ", doselMax[0]: " << doselMax[0] <<
  //                                                                          ", doselMin[1]: " << doselMin[1] << ", doselMax[1]: " << doselMax[1] <<
  //                                                                          ", doselMin[2]: " << doselMin[2] << ", doselMax[2]: " << doselMax[2] << Gateendl);

  //GateMessage("Actor", 10, "[GateVoxelizedMass::" << __FUNCTION__ << "] DEBUG: round(doselMin[0]): " << round(doselMin[0]) << ", round(doselMax[0]): " << round(doselMax[0]) <<
  //                                                                          ", round(doselMin[1]): " << round(doselMin[1]) << ", round(doselMax[1]): " << round(doselMax[1]) <<
  //                                                                          ", round(doselMin[2]): " << round(doselMin[2]) << ", round(doselMax[2]): " << round(doselMax[2]) << Gateendl);

  for(unsigned long int x = round(doselMin[0]); x < round(doselMax[0]); x++)
    for(unsigned long int y = round(doselMin[1]); y < round(doselMax[1]); y++)
      for(unsigned long int z = round(doselMin[2]); z < round(doselMax[2]); z++)
        {
          std::vector<bool> isMin(3,false),isMax(3,false);
          std::vector<unsigned long int>               origCoord(3,-1);
          std::vector<std::vector<unsigned long int> > coord(3);
          std::vector<std::vector<double> >            coef (3);

          origCoord[0] = x;
          origCoord[1] = y;
          origCoord[2] = z;

          //GateMessage("Actor", 10, "[GateVoxelizedMass::" << __FUNCTION__ << "] DEBUG: origCoord[0]: " << origCoord[0] << ", origCoord[1]: " << origCoord[1] << ", origCoord[2]: " << origCoord[2] << Gateendl);

          for(size_t dim=0; dim < 3; dim++)
            {
              if(origCoord[dim]==round(doselMin[dim])&&fmod(doselMin[dim],1)>1e-8)
                isMin[dim]=true;
              if(origCoord[dim]==round(doselMax[dim])-1&&fmod(doselMax[dim],1)>1e-8)
                isMax[dim]=true;

              if(isMin[dim] && isMax[dim])
                {
                  if(fmod(doselMin[dim],1) >= 0.5 && fmod(doselMax[dim],1) < 0.5)
                    {
                      coord[dim].push_back(origCoord[dim]-1);
                      coord[dim].push_back(origCoord[dim]);
                      coord[dim].push_back(origCoord[dim]+1);

                      coef[dim].push_back(1-fmod(doselMin[dim],1));
                      coef[dim].push_back(1);
                      coef[dim].push_back(fmod(doselMax[dim],1));
                    }
                  else if(fmod(doselMin[dim],1)>=0.5)
                    {
                      coord[dim].push_back(origCoord[dim]-1);
                      coord[dim].push_back(origCoord[dim]);

                      coef[dim].push_back(1-fmod(doselMin[dim],1));
                      coef[dim].push_back(fmod(doselMax[dim],1));
                    }
                  else if(fmod(doselMax[dim],1)<0.5)
                    {
                      coord[dim].push_back(origCoord[dim]);
                      coord[dim].push_back(origCoord[dim]+1);

                      coef[dim].push_back(1-fmod(doselMin[dim],1));
                      coef[dim].push_back(fmod(doselMax[dim],1));
                    }
                  else
                    {
                      coord[dim].push_back(origCoord[dim]);
                      coef[dim].push_back(std::abs((1-fmod(doselMin[dim],1))-fmod(doselMax[dim],1))); //FIXME ?
                    }
                }
              else if(isMin[dim])
                {
                  if(fmod(doselMin[dim],1)>=0.5)
                    {
                      coord[dim].push_back(origCoord[dim]-1);
                      coord[dim].push_back(origCoord[dim]);

                      coef[dim].push_back(1-fmod(doselMin[dim],1));
                      coef[dim].push_back(1);
                    }
                  else
                    {
                      coord[dim].push_back(origCoord[dim]);
                      coef[dim].push_back(1-fmod(doselMin[dim],1));
                    }
                }
              else if(isMax[dim])
                {
                  if(fmod(doselMax[dim],1)<0.5)
                    {
                      coord[dim].push_back(origCoord[dim]);
                      coord[dim].push_back(origCoord[dim]+1);

                      coef[dim].push_back(1);
                      coef[dim].push_back(fmod(doselMax[dim],1));
                    }
                  else
                    {
                      coord[dim].push_back(origCoord[dim]);
                      coef[dim].push_back(fmod(doselMax[dim],1));
                    }
                }
              else
                {
                  coord[dim].push_back(origCoord[dim]);
                  coef[dim].push_back(1);
                }
              if(coord[dim].size() != coef[dim].size())
                {
                  GateError("[GateVoxelizedMass::" << __FUNCTION__ << "] ERROR : Size of coord and coef are not the same !" << Gateendl);
                  exit(EXIT_FAILURE);
                }
            }

          for(size_t xVox = 0; xVox < coord[0].size(); xVox++)
            for(size_t yVox = 0; yVox < coord[1].size(); yVox++)
              for(size_t zVox = 0; zVox < coord[2].size(); zVox++)
                {
                  //GateMessage("Actor", 10, "[GateVoxelizedMass::" << __FUNCTION__ << "] DEBUG: coord[0][xVox]: " << coord[0][xVox] << ", coord[1][xVox]: " << coord[1][yVox] << ", coord[2][xVox]: " << coord[2][zVox] << Gateendl);

                  if (mMaterialFilter == "" ||
                      mMaterialFilter == GetVoxelMatName(coord[0][xVox], coord[1][yVox], coord[2][zVox]))
                    {
                      const double coefVox(coef[0][xVox] * coef[1][yVox] * coef[2][zVox]);

                      doselReconstructedVolume      += GetVoxelVolume() * coefVox;
                      doselReconstructedMass[index] += GetVoxelMass(coord[0][xVox], coord[1][yVox], coord[2][zVox]) * coefVox;

                      if(doselReconstructedVolume < 0.)
                        GateError("[GateVoxelizedMass::" << __FUNCTION__ << "] ERROR : doselReconstructedVolume is negative !" << Gateendl
                                  <<"     More informations :" << Gateendl
                                  <<"            doselReconstructedVolume=" << doselReconstructedVolume << Gateendl
                                  <<"            Voxel Volume: " << GetVoxelVolume() << Gateendl
                                  <<"            coefVox=" << coefVox <<Gateendl);

                      if(doselReconstructedMass[index] < 0.)
                        GateError("[GateVoxelizedMass::" << __FUNCTION__ << "] ERROR : doselReconstructedMass is negative !" << Gateendl
                                  <<"     More informations:" << Gateendl
                                  <<"            doselReconstructedMass[" << index << "]=" << doselReconstructedMass[index] << Gateendl
                                  <<"            Voxel Mass: " << GetVoxelMass(coord[0][xVox], coord[1][yVox], coord[2][zVox]) << Gateendl
                                  <<"            coefVox= " << coefVox << Gateendl);
                    }
                }
        }

  if(doselReconstructedMass[index] < 0.)
    GateError("[GateVoxelizedMass::" << __FUNCTION__ << "] ERROR: doselReconstructedMass is negative ! (doselReconstructedMass["<<index<<"] = "<<doselReconstructedMass[index]<<")"<<Gateendl);
  if(doselReconstructedVolume < 0.)
    GateError("[GateVoxelizedMass::" << __FUNCTION__ << "] ERROR: doselReconstructedVolume is negative ! (doselReconstructedVolume = "<<doselReconstructedVolume<<")"<<Gateendl);

  GateMessage("Actor", 10, "[GateVoxelizedMass::" << __FUNCTION__ << "] Ended" << Gateendl);

  return std::make_pair(doselReconstructedMass[index],doselReconstructedVolume);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
pair<double,double> GateVoxelizedMass::VoxelIteration(const G4VPhysicalVolume* motherPV,int Generation,const G4RotationMatrix motherRotation,const G4ThreeVector motherTranslation,int index)
{
  if (Generation == 0)
    {
      if (IsLVParameterized(motherPV->GetLogicalVolume()))
        GateError("The volume " << motherPV->GetName() << " is parameterized !" << Gateendl
                  << "Please attach the DoseActor directly on this volume !" << Gateendl);

      // IMPORTANT WARNING
      if (motherPV->GetLogicalVolume()->GetSolid()->GetEntityType() != "G4Box")
        GateMessage("Actor", 0, "[GateVoxelizedMass::" << __FUNCTION__ << "] WARNING: Attaching a DoseActor to a volume with another geometry than a box can lead to a wrong dose calculation ! Please verify that the volume of the dosel is correctly reconstructed !" << Gateendl);

      GateMessage("Actor", 2, Gateendl << "[GateVoxelizedMass::" << __FUNCTION__ << "] Dosel n°" << index << ":" << Gateendl);

      GateMessage("Actor", 5, "[GateVoxelizedMass::" << __FUNCTION__ << "] DEBUG: Dosel n°" << index << " coordinates: "
                  << mImage->GetVoxelCenterFromIndex(index).x() << ","
                  << mImage->GetVoxelCenterFromIndex(index).y() << ","
                  << mImage->GetVoxelCenterFromIndex(index).z() << Gateendl);

      mFilteredVolumeMass        = 0.;
      mFilteredVolumeCubicVolume = 0.;
      mIsFilteredVolumeProcessed = false;
    }

  GateMessage("Actor", 2, "[GateVoxelizedMass::VoxelIteration] Generation n°" << Generation << " (motherPV: " <<  motherPV->GetName() << ") :" << Gateendl);

  const G4LogicalVolume* motherLV = motherPV->GetLogicalVolume();
  G4VSolid*              motherSV = motherLV->GetSolid()->Clone();

  bool isFilteredVolume(false);
  // FILTERING /////////////////////////////////////////////////////////////
  if (mHasFilter && mVolumeFilter != "" && motherSV->GetName() == mVolumeFilter)
    {
      GateMessage("Actor", 2, "[GateVoxelizedMass::VoxelIteration] VolumeFilter: Filtered volume " <<  motherSV->GetName() << " finded ! (Generation n°" << Generation << ")" << Gateendl);

      isFilteredVolume = true;
    }
  //////////////////////////////////////////////////////////////////////////

  double motherMass(0.);
  double motherProgenyMass(0.);
  double motherProgenyCubicVolume(0.);

  if (motherSV->GetCubicVolume() == 0.)
    GateError("Error: motherSV->GetCubicVolume() is null ! (motherPhysicalVolume : "<<motherPV->GetName()<<")"<<Gateendl);

  // Dosel absolute rotation and translation
  const G4RotationMatrix doselAbsoluteRotation    = mImage->GetTransformMatrix();
  const G4ThreeVector    doselAbsoluteTranslation = mImage->GetVoxelCenterFromIndex(index);

  // Mother absolute rotation and translation
  G4RotationMatrix motherAbsoluteRotation    = motherRotation;
  G4ThreeVector    motherAbsoluteTranslation = motherTranslation;

  if (Generation == 0)
    {
      motherAbsoluteRotation.   set(0.,0.,0.);
      motherAbsoluteTranslation.set(0.,0.,0.);
    }

  // Mother->Dosel relative rotation and translation
  G4RotationMatrix    motherDoselRelativeRotation    = motherAbsoluteRotation    * doselAbsoluteRotation;
  const G4ThreeVector motherDoselRelativeTranslation = motherAbsoluteTranslation - doselAbsoluteTranslation;

  GateMessage("Actor", 2, "[GateVoxelizedMass::VoxelIteration] " << motherPV->GetName() << " (before overlap mother-dosel) : " << G4BestUnit(motherSV->GetCubicVolume(),"Volume") << Gateendl);

  // Overlap Dosel-Mother
  motherSV = new G4IntersectionSolid(motherSV->GetName(),
                                     doselSV,
                                     motherSV,
                                     &motherDoselRelativeRotation,
                                     motherDoselRelativeTranslation);

  // If the mother's doesn't intersects the dosel
  const double ratio     = motherSV->GetCubicVolume() * 100. / doselSV->GetCubicVolume();
  const double tolerance = 1.;

  if (ratio < tolerance)
    {
      GateMessage("Actor", 0, "[GateVoxelizedMass::VoxelIteration] WARNING: " <<  motherPV->GetName() << " is not contained inside the dosel n°" << index << Gateendl
                  << "     diff overlap Dosel-Mother: " << ratio << "%" << Gateendl
                  << "     diff overlap tolerance   : " << tolerance << "%" << Gateendl);

      return make_pair(0.,-1.);
    }

  GateMessage("Actor", 2, "[GateVoxelizedMass::VoxelIteration] "<< motherPV->GetName() <<" (after overlap mother-dosel)  : " << G4BestUnit(motherSV->GetCubicVolume(),"Volume") << Gateendl);

  if (IsLVParameterized(motherLV))
    {
      GateMessage("Actor", 2, "[GateVoxelizedMass::VoxelIteration] WARNING: " << motherPV->GetName() << " is parameterized !" << Gateendl
                  << " ==> Returning null mass ! " << Gateendl);

      return make_pair(0., motherSV->GetCubicVolume());
    }

  // Calculation for daughter(s) ///////////////////////////////////////////
  if(motherLV->GetNoDaughters() > 0)
    {
      for(unsigned int i=0; i<motherLV->GetNoDaughters(); i++)
        {
          const G4VPhysicalVolume*  daughterPV = motherLV->GetDaughter(i);
          G4VSolid*                 daughterSV = daughterPV->GetLogicalVolume()->GetSolid()->Clone();

          // Mother->Daughter relative translation and rotation
          const G4RotationMatrix motherDaughterRelativeRotation    = daughterPV->GetObjectRotationValue();
          const G4ThreeVector    motherDaughterRelativeTranslation = daughterPV->GetObjectTranslation();

          // Mother->GMother relative translation and rotation
          const G4RotationMatrix motherGMotherRelativeRotation    = motherPV->GetObjectRotationValue();
          const G4ThreeVector    motherGMotherRelativeTranslation = motherPV->GetObjectTranslation();

          // Daughter absolute translation and rotation
          const G4RotationMatrix daughterAbsoluteRotation    = motherDaughterRelativeRotation * motherAbsoluteRotation;
          const G4ThreeVector    daughterAbsoluteTranslation = motherGMotherRelativeRotation  * motherDaughterRelativeTranslation + motherAbsoluteTranslation;

          // Dosel->Daughter relative translation and rotation
          G4RotationMatrix       doselDaughterRelativeRotation    = daughterAbsoluteRotation * doselAbsoluteRotation;
          const G4ThreeVector    doselDaughterRelativeTranslation = doselAbsoluteTranslation - daughterAbsoluteTranslation;

          const double motherCubicVolumeBefore = motherSV->GetCubicVolume();

          GateMessage("Actor", 2, "[GateVoxelizedMass::VoxelIteration] "<<motherPV->GetName()<<" (before sub. mother-daughter) : " << G4BestUnit(motherSV->GetCubicVolume(),"Volume") << " (daughterPV : "<< daughterPV->GetName() <<")"<< Gateendl);

          if(motherCubicVolumeBefore > 0.)
            {
              // Creating a backup of motherSV before the substraction
              G4VSolid* motherBeforeSubSV = motherSV->Clone();

              // Substraction Dosel-Daughter
              motherSV = new G4SubtractionSolid(motherSV->GetName(),
                                                motherSV,
                                                daughterSV,
                                                &doselDaughterRelativeRotation,
                                                doselDaughterRelativeTranslation);

              GateMessage("Actor", 2, "[GateVoxelizedMass::VoxelIteration] "<<motherPV->GetName()<<" (after sub. mother-daughter)  : " << G4BestUnit(motherSV->GetCubicVolume(),"Volume") << " (daughterPV : "<<daughterPV->GetName()<<")"<< Gateendl);


              const double diff((motherSV->GetCubicVolume() - motherCubicVolumeBefore) *100. / motherCubicVolumeBefore);
              double substractionError(1.);

              if(diff > substractionError)
                {
                  GateMessage("Actor", 0, "[GateVoxelizedMass] WARNING: " << daughterPV->GetName() << " seems to be outside the dosel n°" << index << " !" << Gateendl);

                  GateMessage("Actor", 1, " => Volume of " << motherPV->GetName() << " after substraction with " << daughterPV->GetName() << " is bigger than before !" << Gateendl
                              << "     diff substraction Mother-Daughter: " << diff << "%" << Gateendl
                              << "     diff substraction tolerance      : ±" << substractionError << "%" << Gateendl);
                  GateMessage("Actor", 0, " ===> " << daughterPV->GetName() << " ignored for the dosel n°" << index << " !" << Gateendl);

                  motherSV = motherBeforeSubSV;
                }
              else if (diff < -substractionError)
                {
                  if (!isFilteredVolume) // If filtered volume no need to process daughter's daugthers
                    {
                      const pair<double,double> daughterIteration(VoxelIteration(daughterPV,Generation+1,daughterAbsoluteRotation,daughterAbsoluteTranslation,index));
                      double daughterCubicVolume = daughterIteration.second;

                      if (daughterIteration.first == 0. && !mHasFilter)
                        GateMessage("Actor", 2, "[GateVoxelizedMass] WARNING: daughterIteration.first (mass) is null ! (daughterPV: " << daughterPV->GetName() << ")" << Gateendl
                                    << " => Maybe " << daughterPV->GetName() << " is (partially) outside " << motherPV->GetName() << "." << Gateendl);


                      if (daughterCubicVolume == -1.)
                        {
                          GateMessage("Actor", 0, "[GateVoxelizedMass] WARNING: " << daughterPV->GetName() << " seems to be outside the dosel n°" << index << " !" << Gateendl);
                          GateMessage("Actor", 1, " => GEANT4 has trouble to compute soustraction between " << daughterPV->GetName() << " and its mother " << motherPV->GetName() << " !" << Gateendl);
                          GateMessage("Actor", 1, " => It can be related to the geometry of " << daughterPV->GetName() << " (" << daughterSV->GetEntityType() << ")" << Gateendl
                                      << "     diff substraction Mother-Daughter: " << diff << "%" << Gateendl
                                      << "     diff substraction tolerance      : ±" << substractionError << "%" << Gateendl);
                          GateMessage("Actor", 0, " ===> " << daughterPV->GetName() << " ignored for the dosel n°" << index << " !" << Gateendl);

                          motherSV = motherBeforeSubSV;
                        }
                      else if (daughterCubicVolume == 0.)
                        GateError("ERROR: " << daughterPV->GetName() << " cubic volume is null in the dosel n°" << index << " !" << Gateendl
                                  << "  => Maybe " << daughterPV->GetName() << " is (partially) outside " << motherPV->GetName() << "." << Gateendl
                                  << "     diff substraction Mother-Daughter: " << diff << "%" << Gateendl
                                  << "     diff substraction tolerance      : ±" << substractionError << "%" << Gateendl);

                      const double diffVol((daughterCubicVolume-(motherBeforeSubSV->GetCubicVolume()-motherSV->GetCubicVolume()))*100./(motherBeforeSubSV->GetCubicVolume()-motherSV->GetCubicVolume()));
                      const double diffVolTolerance(1.);

                      if (abs(diffVol) > diffVolTolerance)
                        {
                          GateMessage("Actor", 0, "[GateVoxelizedMass] WARNING: Volume of " << daughterPV->GetName() << " is not correctly calculated in the dosel n°" << index << " !" << Gateendl);

                          GateMessage("Actor", 1, "Informations:" << Gateendl
                                      << "     Daughter volume (theorical)    : " << G4BestUnit(motherBeforeSubSV->GetCubicVolume()-motherSV->GetCubicVolume(),"Volume") << Gateendl
                                      << "     Daughter volume (reconstructed): " << G4BestUnit(daughterCubicVolume,"Volume") << Gateendl
                                      << "     diff                           : " << diffVol << "%" << Gateendl
                                      << "     diff tolerance                 : ±" << diffVolTolerance << "%" << Gateendl);
                          GateMessage("Actor", 0, " ===> Using theorical volume for " << daughterPV->GetName()<< Gateendl);
                          daughterCubicVolume = motherBeforeSubSV->GetCubicVolume() - motherSV->GetCubicVolume();
                        }

                      motherProgenyMass        += daughterIteration.first;
                      motherProgenyCubicVolume += daughterCubicVolume;
                    }
                  else
                    {
                      motherProgenyMass        += 0.;
                      motherProgenyCubicVolume += motherBeforeSubSV->GetCubicVolume()-motherSV->GetCubicVolume();
                    }
                }
              else if (abs(diff) <= substractionError)
                {
                  GateMessage("Actor", 2, "[GateVoxelizedMass::VoxelIteration] " << daughterPV->GetName() << " is not contained inside dosel n°" << index << Gateendl);

                  motherSV=motherBeforeSubSV;
                }

              GateMessage("Actor", 2, "[GateVoxelizedMass::VoxelIteration] " << motherPV->GetName() << " volume: " << G4BestUnit(motherSV->GetCubicVolume(),"Volume") << Gateendl);
            }
        }
    }
  //////////////////////////////////////////////////////////////////////////

  // Mother mass & volume //////////////////////////////////////////////////
  const double motherCubicVolume = motherSV->GetCubicVolume();
  const double motherDensity     = motherLV->GetMaterial()->GetDensity();

  motherMass        =  motherCubicVolume * motherDensity;
  motherProgenyMass += motherMass;
  motherProgenyCubicVolume += motherCubicVolume;

  GateMessage("Actor", 2, "[GateVoxelizedMass::VoxelIteration] "<<motherPV->GetName()<<" density :" << G4BestUnit(motherDensity,"Volumic Mass")<< Gateendl);
  //////////////////////////////////////////////////////////////////////////

  // Saving ////////////////////////////////////////////////////////////////
  mCubicVolume[index].push_back(make_pair(motherSV->GetName(),motherCubicVolume));
  mMass[index].push_back       (make_pair(motherSV->GetName(),motherMass));
  //////////////////////////////////////////////////////////////////////////

  // FILTERING /////////////////////////////////////////////////////////////
  if (isFilteredVolume)
    {
      mFilteredVolumeMass        = motherMass;
      mFilteredVolumeCubicVolume = motherCubicVolume;
      mIsFilteredVolumeProcessed = true;
    }
  //////////////////////////////////////////////////////////////////////////

  if (motherProgenyMass == 0.)
    {
      GateMessage("Actor", 0, "[GateVoxelizedMass::VoxelIteration] SV of index " << index << ", generation " << Generation << Gateendl);
      GateMessage("Actor", 0, "[GateVoxelizedMass::VoxelIteration] Is Filtered Volume Processed: " << mIsFilteredVolumeProcessed << Gateendl);
      //for(int i = 0;i<motherLV->GetNoDaughters();i++)
      //  GateMessage("Actor", 0, "             * DaughterName: " << motherLV->GetDaughter(i)->GetName() << Gateendl);

      for(size_t i = 0;i<mMass[index].size();i++)
        GateMessage("Actor", 0, "             * Name: " << mMass[index][i].first << ", Mass: " << G4BestUnit(mMass[index][i].second,"Mass") << ", Volume: " << G4BestUnit(mCubicVolume[index][i].second,"Volume") << Gateendl);

      GateError("Error: motherProgenyMass is null ! (index: " << index << ", motherPhysicalVolume : " << motherPV->GetName() << ")" << Gateendl);
    }

  if(motherProgenyCubicVolume == 0.)
    GateError("Error: motherProgenyCubicVolume is null ! (index: " << index << ", motherPhysicalVolume : " << motherPV->GetName() << ") " << Gateendl);

  if (Generation == 0)
    {
      double diff(0.);
      double substractionError(1.);

      if (motherPV->GetLogicalVolume()->GetSolid()->GetEntityType() == "G4Box")
        diff = (motherProgenyCubicVolume - doselSV->GetCubicVolume()) * 100. / doselSV->GetCubicVolume();
      else
        GateMessage("Actor", 0, "[GateVoxelizedMass::VoxelIteration] WARNING: The volume attached to this DoseActor is a " << motherPV->GetLogicalVolume()->GetSolid()->GetEntityType() << ". The reconstruted volume of the dosel is " << G4BestUnit(motherProgenyCubicVolume,"Volume") << ". Please verify it is correct !" << Gateendl);


      if (abs(diff) > substractionError)
        GateError("Error: Dosel n°" << index << " is wrongly reconstructed !" << Gateendl <<
                  "                            SV geometry type      = " << motherPV->GetLogicalVolume()->GetSolid()->GetEntityType() << Gateendl <<
                  "                            dosel (theorical)     = " << G4BestUnit(doselSV->GetCubicVolume(),"Volume") << Gateendl <<
                  "                            dosel (reconstructed) = " << G4BestUnit(motherProgenyCubicVolume,"Volume") << Gateendl <<
                  "                            difference            = " << diff << "%" << Gateendl <<
                  "                            difference tolerance  = ±" << substractionError << "%" <<Gateendl );

      GateMessage("Actor", 2, "[GateVoxelizedMass::VoxelIteration] Dosel n°"<< index << " informations :" << Gateendl <<
                  "                                     motherProgenyMass        = " << G4BestUnit(motherProgenyMass,"Mass") << Gateendl <<
                  "                                     motherProgenyCubicVolume = " << G4BestUnit(motherProgenyCubicVolume,"Volume") << Gateendl);

      if (mHasFilter && mVolumeFilter != "") {
        if (mIsFilteredVolumeProcessed)
          return make_pair(mFilteredVolumeMass,mFilteredVolumeCubicVolume);
        else
          return make_pair(0.,0.);
      }
    }

  return make_pair(motherProgenyMass,motherProgenyCubicVolume);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateVoxelizedMass::GetPartialVolumeWithSV(int index,G4String SVName)
{
  if (mIsParameterised) {
    GateError("Error: GateVoxelizedMass::GetPartialVolumeWithSV: This method doesn't work with voxelized volumes !"<<Gateendl);
    exit(EXIT_FAILURE);
  }

  if(mCubicVolume[index].empty())
    GetDoselMass(index);

  for(size_t i=0;i<mCubicVolume[index].size();i++)
    if(mCubicVolume[index][i].first==SVName)
      return mCubicVolume[index][i].second;

  GateError("!!! ERROR : GateVoxelizedMass::GetPartialVolume : Can't find "<<SVName<<" inside the dosel n°"<<index<<" !"<<Gateendl);
  return -1.;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateVoxelizedMass::GetPartialMassWithSV(int index,G4String SVName)
{
  if (mIsParameterised) {
    GateError("Error: GateVoxelizedMass::GetPartialMassWithSV: This method doesn't work with voxelized volumes !"<<Gateendl);
    exit(EXIT_FAILURE);
  }

  if(mMass[index].empty())
    GetDoselMass(index);

  for(size_t i=0;i<mMass[index].size();i++)
    if(mMass[index][i].first==SVName)
      return mMass[index][i].second;

  GateError("!!! ERROR : GateVoxelizedMass::GetPartialMass : Can't find "<<SVName<<" inside the dosel n°"<<index<<" !"<<Gateendl);
  return -1.;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//double GateVoxelizedMass::GetPartialVolumeWithMatName(int index)
//{
//  if(mMaterialFilter=="")
//    GateError("Error: GateVoxelizedMass::GetPartialVolumeWithMatName: No material filter defined !"<<Gateendl);
//
//  if(!mIsParameterised)
//    GateError("Error: GateVoxelizedMass::GetPartialVolumeWithMatName: This method only work with voxelized volumes !"<<Gateendl);
//
//  if(!mIsVecGenerated)
//    GenerateVectors();
//
//  return;
//}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateVoxelizedMass::GetPartialMassWithMatName(int index)
{
  return GetDoselMass(index);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
int GateVoxelizedMass::GetNumberOfVolumes(int index)
{
  return mMass[index].size();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateVoxelizedMass::GetTotalVolume()
{
  return mImage->GetVoxelVolume();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVoxelizedMass::SetEdep(int index,G4String SVName,double edep)
{
  bool ok(false);
  for(size_t i=0;i<mEdep[index].size();i++)
    if(mEdep[index][i].first==SVName)
      {
        ok=true;
        mEdep[index][i].second+=edep;
      }

  if(!ok)
    mEdep[index].push_back(std::make_pair(SVName,edep));
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateVoxelizedMass::GetMaxDose(int index)
{
  double edepmax(0.);
  G4String SVName("");

  for(size_t i=0;i<mEdep[index].size();i++)
    if(mEdep[index][i].second>edepmax)
      {
        SVName=mEdep[index][i].first;
        edepmax=mEdep[index][i].second;
      }

  return edepmax/GetPartialMassWithSV(index,SVName);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVoxelizedMass::UpdateImage(GateImageDouble* image)
{
  if (mIsInitialized) {
    const std::vector<double> vector = GetDoselMassVector();

    for (size_t i=0; i<vector.size(); i++)
      image->AddValue(i, vector[i]/kg);
  }
  else
    GateMessage("Actor", 0, "[GateVoxelizedMass::" << __FUNCTION__ << "] WARNING: VoxelizedMass is not initialized. Image update ignored !" << Gateendl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVoxelizedMass::SetMaterialFilter(G4String MatName)
{
  if (MatName != "") {
    if (mHasFilter)
      GateError("[GateVoxelizedMass::" << __FUNCTION__ << "] ERROR: material filter is not compatible with other filters !" << Gateendl);
    else if (mHasExternalMassImage)
      GateError("[GateVoxelizedMass::" << __FUNCTION__ << "] ERROR: mass image importation is not compatible with filters !" << Gateendl);
    else {
      mMaterialFilter = MatName;
      mHasFilter = true;
    }
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVoxelizedMass::SetVolumeFilter(G4String VolName)
{
  if (VolName == "world")
    GateError( "[GateVoxelizedMass::" << __FUNCTION__ << "] ERROR: DoseActor doesn't work when attached to world !" << Gateendl);
  if (VolName != "") {
    if (mHasFilter)
      GateError( "[GateVoxelizedMass::" << __FUNCTION__ << "] ERROR: volume filter is not compatible with other filters !" << Gateendl);
    //else if (mHasExternalMassImage)
    //  GateError( "Error: GateVoxelizedMass::SetVolumeFilter: mass image importation is not compatible with filters !" << Gateendl);
    else {
      mVolumeFilter=VolName+"_solid";
      mHasFilter=true;
    }
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVoxelizedMass::SetExternalMassImage(G4String extMassFile)
{
  if(extMassFile != "")
    {
      mMassFile=extMassFile;
      mHasExternalMassImage=true;

      //if(mHasFilter)
      //  GateError( "Error: GateVoxelizedMass::SetExternalMassImage: mass image importation is not compatible with filters !" << Gateendl);
    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateImage* Copy(GateImage* oldImage)
{
  GateImage* newImage = new GateImage;

  newImage->SetResolutionAndVoxelSize(oldImage->GetResolution(),oldImage->GetVoxelSize());
  newImage->SetOrigin(oldImage->GetOrigin());
  newImage->SetTransformMatrix(oldImage->GetTransformMatrix());
  newImage->Allocate();
  newImage->SetOutsideValue(oldImage->GetMinValue()-1);
  newImage->Fill(oldImage->GetOutsideValue());

  for (signed long int index = 0; index < oldImage->GetNumberOfValues(); index++)
    newImage->SetValue(index,oldImage->GetValue(index));

  return newImage;
}
//-----------------------------------------------------------------------------
