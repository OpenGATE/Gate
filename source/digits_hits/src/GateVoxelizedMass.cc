/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
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
  mIsInitialized        =false;
  mIsParameterised      =false;
  mIsVecGenerated       =false;
  mHasFilter            =false;
  mHasExternalMassImage =false;

  mMassFile       ="";
  mMaterialFilter ="";
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVoxelizedMass::Initialize(const G4String mExtVolumeName, const GateImageDouble mExtImage)
{
  mVolumeName=mExtVolumeName;
  mImage=mExtImage;

  mIsParameterised=false;
  mIsVecGenerated=false;

  mCubicVolume.resize(mImage.GetNumberOfValues());
  mMass.resize(mImage.GetNumberOfValues());
  mEdep.resize(mImage.GetNumberOfValues());
  for(int index=0 ; index<mImage.GetNumberOfValues() ; index++)
  {
    mCubicVolume[index].clear();
    mMass[index].clear();
    mEdep[index].clear();
  }

  doselExternalMass.clear();
  if(mHasExternalMassImage && mMassFile!="")
  {
    mMassImage.Read(mMassFile);

    if (mMassImage.GetResolution() != mImage.GetResolution())
      GateError("Error: " << mMassFile << " hasn't the right resolution !" << Gateendl <<
                "   Actor resolution: " << mImage.GetResolution() << Gateendl <<
                "   " << mMassFile << " resolution: " << mMassImage.GetResolution() << Gateendl);

    if (mMassImage.GetNumberOfValues() != mImage.GetNumberOfValues())
      GateError("Error: " << mMassFile << " hasn't the right number of dosels !" << Gateendl <<
                "   Actor number of dosels: " << mImage.GetNumberOfValues() << Gateendl <<
                "   " << mMassFile << " number of dosels: " << mMassImage.GetNumberOfValues() << Gateendl);

    double diff((mMassImage.GetVoxelVolume()-mImage.GetVoxelVolume())*100/mImage.GetVoxelVolume());
    double substractionError(0.5);

    if (std::abs(diff) > substractionError)
      GateError("Error: " << mMassFile << " hasn't the right dosel cubic volume !" << Gateendl <<
                "   Actor dosel cubic volume: " << G4BestUnit(mImage.GetVoxelVolume(),"Volume") << Gateendl <<
                "   " << mMassFile << " dosel cubic volume: " << G4BestUnit(mMassImage.GetVoxelVolume(),"Volume") << Gateendl <<
                "   Difference: " << diff << Gateendl);

    doselExternalMass.resize(mMassImage.GetNumberOfValues(),-1.);
    for(int i=0;i<mMassImage.GetNumberOfValues();i++)
      doselExternalMass[i]=mMassImage.GetValue(i)*kg;
  }

  DAPV=G4PhysicalVolumeStore::GetInstance()->GetVolume(mVolumeName+"_phys");
  DALV=DAPV->GetLogicalVolume();

  vectorSV.clear();
  doselReconstructedMass.clear();
  doselReconstructedCubicVolume.clear();
  doselReconstructedMass.resize(mImage.GetNumberOfValues(),0.);
  doselReconstructedCubicVolume.resize(mImage.GetNumberOfValues(),0.);

  doselSV=new G4Box("DoselSV",
                    mImage.GetVoxelSize().getX()/2.0,
                    mImage.GetVoxelSize().getY()/2.0,
                    mImage.GetVoxelSize().getZ()/2.0);

  if (IsLVParameterized(DALV)) {
    imageVolume=dynamic_cast<GateVImageVolume*>(GateObjectStore::GetInstance()->FindVolumeCreator(DAPV));

    if(doselExternalMass.size()==0)
      GateVoxelizedMass::GenerateVoxels();

    mIsParameterised=true;
  }

  if (mHasFilter) {
    if (mIsParameterised && mVolumeFilter != "")
      GateError( "Error: GateVoxelizedMass::Initialize: volume filter is not compatible with parameterised volumes !" << Gateendl);
    else if (!mIsParameterised && mMaterialFilter != "")
      GateError( "Error: GateVoxelizedMass::Initialize: material filter is only compatible with parameterised volumes !" << Gateendl);
  }

  mIsInitialized=true;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool GateVoxelizedMass::IsLVParameterized(const G4LogicalVolume* LV)
{
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
double GateVoxelizedMass::GetVoxelMass(const int index)
{
  // Case of imported mass image
  if(mHasExternalMassImage)
  {
    if(doselExternalMass.size()>0)
      return doselExternalMass[index];
    else
      GateError( "Error: GateVoxelizedMass::GetVoxelMass: initialization of doselExternalMass is incorrect !" << Gateendl);
  }

  if(doselReconstructedMass[index]==-1.)
  {
    if(mIsParameterised)
      doselReconstructedData=ParameterizedVolume(index);
    else
      doselReconstructedData=VoxelIteration(DAPV,0,DAPV->GetObjectRotationValue(),DAPV->GetObjectTranslation(),index);

    doselReconstructedMass[index]=doselReconstructedData.first;
    doselReconstructedCubicVolume[index]=doselReconstructedData.second;
  }

  return doselReconstructedMass[index];
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
std::vector<double> GateVoxelizedMass::GetVoxelMassVector()
{
  if(doselExternalMass.size()>0)
    return doselExternalMass;

  if(!mIsVecGenerated) GenerateVectors();

  return doselReconstructedMass;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVoxelizedMass::GenerateVectors()
{
  time_t timer1,timer2,timer3,timer4;
  time(&timer1);

  GateMessage("Actor", 0,  "[GateVoxelizedMass] Total voxelized mass calculation for in progress, please wait ... " << Gateendl);

  doselReconstructedTotalCubicVolume=0.;
  doselReconstructedTotalMass=0.;

  GateMessage("Actor", 1, "[GateVoxelizedMass] Number of values in the images: " <<  mImage.GetNumberOfValues() << Gateendl);
  GateMessage("Actor", 1, "[GateVoxelizedMass] Is parameterised ? " <<  mIsParameterised << Gateendl);

  for(long int i=0;i<mImage.GetNumberOfValues();i++)
  {
    time(&timer3);

    if(mIsParameterised)
      doselReconstructedData=ParameterizedVolume(i);
    else
      doselReconstructedData=VoxelIteration(DAPV,0,DAPV->GetObjectRotationValue(),DAPV->GetObjectTranslation(),i);

    doselReconstructedMass[i]=doselReconstructedData.first;
    doselReconstructedCubicVolume[i]=doselReconstructedData.second;

    doselReconstructedTotalMass+=doselReconstructedMass[i];
    doselReconstructedTotalCubicVolume+=doselReconstructedCubicVolume[i];

    time(&timer4);
    seconds=difftime(timer4,timer1);

    if(difftime(timer4,timer1)>=60&&i%100==0)
    {
      std::cout<<" "<<i*100/mImage.GetNumberOfValues()<<"% (time elapsed : "<<seconds/60<<"min"<<seconds%60<<"s)      \r"<<std::flush;
      // Experimental
      /*seconds=(mImage.GetNumberOfValues()-i)*difftime(timer4,timer3);
      if(seconds!=0.) std::cout<<"Estimated remaining time : "<<seconds/60<<"min"<<seconds%60<<"s ("<<seconds<<"s)                \r"<<std::flush;*/
    }
  }

  time(&timer2);
  seconds=difftime(timer2,timer1);

  G4String s="\n";
  if (mIsParameterised)
    s = "\tNumber of voxels : "+DoubletoString(imageVolume->GetImage()->GetNumberOfValues())+'\n';
  GateMessage("Actor", 1,
              "[GateVoxelizedMass] Summary: mass calculation for voxelized volume :" << G4endl
              <<"\tTime elapsed : " << seconds/60 << "min" << seconds%60 << "s (" << seconds << "s)" << G4endl
              << s
              << "\tNumber of dosels : " << mImage.GetNumberOfValues() << G4endl
              << "\tDosels reconstructed total mass : " << G4BestUnit(doselReconstructedTotalMass,"Mass") << G4endl
              << "\tDosels reconstructed total cubic volume : "
              << G4BestUnit(doselReconstructedTotalCubicVolume,"Volume") << G4endl);
  mIsVecGenerated=true;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVoxelizedMass::GenerateVoxels()
{
  const int nxVoxel=imageVolume->GetImage()->GetResolution().x(),
            nyVoxel=imageVolume->GetImage()->GetResolution().y(),
            nzVoxel=imageVolume->GetImage()->GetResolution().z();

  DABox=(G4Box*)DALV->GetSolid();

  const int nxDosel=round(DABox->GetXHalfLength()/(mImage.GetVoxelSize().x()/2.)),
            nyDosel=round(DABox->GetYHalfLength()/(mImage.GetVoxelSize().y()/2.)),
            nzDosel=round(DABox->GetZHalfLength()/(mImage.GetVoxelSize().z()/2.));

  //G4cout<<"nxVoxel="<<nxVoxel<<G4endl;
  //G4cout<<"nyVoxel="<<nyVoxel<<G4endl;
  //G4cout<<"nzVoxel="<<nzVoxel<<G4endl;

  //G4cout<<"nxDosel="<<nxDosel<<G4endl;
  //G4cout<<"nyDosel="<<nyDosel<<G4endl;
  //G4cout<<"nzDosel="<<nzDosel<<G4endl;

  if(nxDosel>nxVoxel||nyDosel>nyVoxel||nzDosel>nzVoxel)
      GateError("!!! ERROR : The dosel resolution is smaller than the voxel resolution !!!"<<Gateendl);

  voxelMass.resize(nxVoxel);
  voxelMatName.resize(nxVoxel);
  for(int x=0;x<nxVoxel;x++)
  {
    voxelMass[x].resize(nyVoxel);
    voxelMatName[x].resize(nyVoxel);
    for(int y=0;y<nyVoxel;y++)
    {
      voxelMass[x][y].resize(nzVoxel,-1.);
      voxelMatName[x][y].resize(nzVoxel,"");
    }
  }

  voxelCubicVolume=imageVolume->GetImage()->GetVoxelVolume();
  if(voxelCubicVolume<=0.)
    GateError("!!! ERROR : Voxels cubic volume is less or equal to zero !!! (cubic volume="<<voxelCubicVolume<<")"<<Gateendl);

  for(signed long int i=0;i<imageVolume->GetImage()->GetNumberOfValues();i++)
  {
    const int xVoxel(round((DABox->GetXHalfLength()+imageVolume->GetImage()->GetVoxelCenterFromIndex(i).x()-imageVolume->GetImage()->GetVoxelSize().x()/2.)/imageVolume->GetImage()->GetVoxelSize().x())),
              yVoxel(round((DABox->GetYHalfLength()+imageVolume->GetImage()->GetVoxelCenterFromIndex(i).y()-imageVolume->GetImage()->GetVoxelSize().y()/2.)/imageVolume->GetImage()->GetVoxelSize().y())),
              zVoxel(round((DABox->GetZHalfLength()+imageVolume->GetImage()->GetVoxelCenterFromIndex(i).z()-imageVolume->GetImage()->GetVoxelSize().z()/2.)/imageVolume->GetImage()->GetVoxelSize().z()));

    //G4cout<<G4endl;
    //G4cout<<"Index="<<i<<G4endl;
    //G4cout<<"xVoxel="<<xVoxel<<G4endl;
    //G4cout<<"yVoxel="<<yVoxel<<G4endl;
    //G4cout<<"zVoxel="<<zVoxel<<G4endl;

    if(xVoxel>=nxVoxel||yVoxel>=nyVoxel||zVoxel>=nzVoxel)
      GateError("!!! ERROR : Too many voxels !!! (xVoxel="<<xVoxel<<",yVoxel="<<yVoxel<<",zVoxel="<<zVoxel<<")"<<Gateendl);

    //G4cout<<"imageVolume->GetImage()->GetValue(i)="<<imageVolume->GetImage()->GetValue(i)<<G4endl;
    //G4cout<<"imageVolume->GetMaterialNameFromLabel((LabelType)imageVolume->GetImage()->GetValue(i))="<<imageVolume->GetMaterialNameFromLabel(imageVolume->GetImage()->GetValue(i))<<G4endl;

    voxelMatName[xVoxel][yVoxel][zVoxel]=(G4String)imageVolume->GetMaterialNameFromLabel(imageVolume->GetImage()->GetValue(i));

    double density(theMaterialDatabase.GetMaterial(voxelMatName[xVoxel][yVoxel][zVoxel])->GetDensity());

    voxelMass[xVoxel][yVoxel][zVoxel]=density*voxelCubicVolume;

    if(voxelMass[xVoxel][yVoxel][zVoxel]<=0.)
      GateError("!!! ERROR : Voxel (index : "<<i<<") mass is less or equal to zero !!! (mass="<<voxelMass[xVoxel][yVoxel][zVoxel]<<")"<<Gateendl);
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVoxelizedMass::GenerateDosels(const int index)
{
  // INFO : Dimension of the vectors : x = 0, y = 1, z = 2

  doselMin.resize(3,-1.);
  doselMin[0]=(DABox->GetXHalfLength()+mImage.GetVoxelCenterFromIndex(index).getX()-mImage.GetVoxelSize().getX()/2.0)/imageVolume->GetImage()->GetVoxelSize().x();
  doselMin[1]=(DABox->GetYHalfLength()+mImage.GetVoxelCenterFromIndex(index).getY()-mImage.GetVoxelSize().getY()/2.0)/imageVolume->GetImage()->GetVoxelSize().y();
  doselMin[2]=(DABox->GetZHalfLength()+mImage.GetVoxelCenterFromIndex(index).getZ()-mImage.GetVoxelSize().getZ()/2.0)/imageVolume->GetImage()->GetVoxelSize().z();

  doselMax.resize(3,-1.);
  doselMax[0]=(DABox->GetXHalfLength()+mImage.GetVoxelCenterFromIndex(index).getX()+mImage.GetVoxelSize().getX()/2.0)/imageVolume->GetImage()->GetVoxelSize().x();
  doselMax[1]=(DABox->GetYHalfLength()+mImage.GetVoxelCenterFromIndex(index).getY()+mImage.GetVoxelSize().getY()/2.0)/imageVolume->GetImage()->GetVoxelSize().y();
  doselMax[2]=(DABox->GetZHalfLength()+mImage.GetVoxelCenterFromIndex(index).getZ()+mImage.GetVoxelSize().getZ()/2.0)/imageVolume->GetImage()->GetVoxelSize().z();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
std::pair<double,double> GateVoxelizedMass::ParameterizedVolume(const int index)
{
  GenerateDosels(index);

  for(int x=round(doselMin[0]);x<round(doselMax[0]);x++)
    for(int y=round(doselMin[1]);y<round(doselMax[1]);y++)
      for(int z=round(doselMin[2]);z<round(doselMax[2]);z++)
      {
        std::vector<bool> isMin(3,false),isMax(3,false);
        std::vector<int>  origCoord(3,-1);
        std::vector<std::vector<int> >    coord(3);
        std::vector<std::vector<double> > coef(3);

        origCoord[0]=x; origCoord[1]=y; origCoord[2]=z;

        for(int dim=0;dim<3;dim++)
        {
          if(origCoord[dim]==round(doselMin[dim])&&fmod(doselMin[dim],1)>1e-8)
            isMin[dim]=true;
          if(origCoord[dim]==round(doselMax[dim])-1&&fmod(doselMax[dim],1)>1e-8)
            isMax[dim]=true;

          if(isMin[dim]&&isMax[dim])
          {
            if(fmod(doselMin[dim],1)>=0.5&&fmod(doselMax[dim],1)<0.5)
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
          if(coord[dim].size()!=coef[dim].size())
            GateError("!!! ERROR : Size of coord and coef are not the same !"<<Gateendl);
        }

        for(size_t xVox=0;xVox<coord[0].size();xVox++)
          for(size_t yVox=0;yVox<coord[1].size();yVox++)
            for(size_t zVox=0;zVox<coord[2].size();zVox++)
              if(mMaterialFilter==""||mMaterialFilter==voxelMatName[coord[0][xVox]][coord[1][yVox]][coord[2][zVox]])
              {
                double coefVox(coef[0][xVox]*coef[1][yVox]*coef[2][zVox]);

                doselReconstructedCubicVolume[index]+=voxelCubicVolume*coefVox;
                doselReconstructedMass[index]+=voxelMass[coord[0][xVox]][coord[1][yVox]][coord[2][zVox]]*coefVox;

                if(doselReconstructedCubicVolume[index]<0.)
                  GateError("!!! ERROR : doselReconstructedCubicVolume is negative !"<<Gateendl
                          <<"     More informations :"<<Gateendl
                          <<"            doselReconstructedCubicVolume["<<index<<"]="<<doselReconstructedCubicVolume[index]<<Gateendl
                          <<"            voxelCubicVolume="<<voxelCubicVolume<<Gateendl
                          <<"            coefVox="<<coefVox<<Gateendl);

                if(doselReconstructedMass[index]<0.)
                  GateError("!!! ERROR : doselReconstructedMass is negative !"<<Gateendl
                          <<"     More informations :"<<Gateendl
                          <<"            doselReconstructedMass["<<index<<"]="<<doselReconstructedMass[index]<<Gateendl
                          <<"            voxelMass="<<voxelMass[coord[0][xVox]][coord[1][yVox]][coord[2][zVox]]<<Gateendl
                          <<"            coefVox="<<coefVox<<Gateendl);
              }
      }
  if(doselReconstructedMass[index]<0.)
    GateError("!!! ERROR : doselReconstructedMass is negative ! (doselReconstructedMass["<<index<<"]="<<doselReconstructedMass[index]<<")"<<Gateendl);
  if(doselReconstructedCubicVolume[index]<0.)
    GateError("!!! ERROR : doselReconstructedCubicVolume is negative ! (doselReconstructedCubicVolume["<<index<<"]="<<doselReconstructedCubicVolume[index]<<")"<<Gateendl);

  return std::make_pair(doselReconstructedMass[index],doselReconstructedCubicVolume[index]);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
std::pair<double,double> GateVoxelizedMass::VoxelIteration(G4VPhysicalVolume* motherPV,const int Generation,G4RotationMatrix motherRotation,G4ThreeVector motherTranslation,const int index)
{
  if (Generation==0) {
    if (IsLVParameterized(motherPV->GetLogicalVolume()))
      GateError("The volume " << motherPV->GetName() << " is parameterized !" << Gateendl
          << "Please attach the DoseActor directly on this volume !" << Gateendl);

    GateMessage("Actor", 2, Gateendl << "[GateVoxelizedMass::VoxelIteration] Dosel n°" << index << ":" << Gateendl);

    mFilteredVolumeMass = 0.;
    mFilteredVolumeCubicVolume = 0.;
    mIsFilteredVolumeProcessed = false;
  }

  GateMessage("Actor", 2, "[GateVoxelizedMass::VoxelIteration] Generation n°" << Generation << " (motherPV: " <<  motherPV->GetName() << ") :" << Gateendl);

  G4LogicalVolume* motherLV(motherPV->GetLogicalVolume());
  G4VSolid*        motherSV(motherLV->GetSolid());

  double motherMass(0.);
  double motherProgenyMass(0.);
  double motherProgenyCubicVolume(0.);

  if(motherSV->GetCubicVolume()==0.)
    GateError("Error: motherSV->GetCubicVolume() is null ! (motherPhysicalVolume : "<<motherPV->GetName()<<")"<<Gateendl);

  // Dosel absolute rotation and translation
  G4RotationMatrix doselAbsoluteRotation    = mImage.GetTransformMatrix();
  G4ThreeVector    doselAbsoluteTranslation = mImage.GetVoxelCenterFromIndex(index);

  // Mother absolute rotation and translation
  G4RotationMatrix motherAbsoluteRotation    = motherRotation;
  G4ThreeVector    motherAbsoluteTranslation = motherTranslation;

  if(Generation==0) {
    motherAbsoluteRotation.   set(0.,0.,0.);
    motherAbsoluteTranslation.set(0.,0.,0.);
  }

  // Mother->Dosel relative rotation and translation
  G4RotationMatrix motherDoselRelativeRotation    = motherAbsoluteRotation    * doselAbsoluteRotation;
  G4ThreeVector    motherDoselRelativeTranslation = motherAbsoluteTranslation - doselAbsoluteTranslation;

  GateMessage("Actor", 2, "[GateVoxelizedMass::VoxelIteration] " << motherPV->GetName() << " (before overlap mother-dosel) : " << G4BestUnit(motherSV->GetCubicVolume(),"Volume") << Gateendl);

  // Overlap Dosel-Mother
  motherSV=new G4IntersectionSolid(motherSV->GetName(),
                                  doselSV,
                                  motherSV,
                                  &motherDoselRelativeRotation,
                                  motherDoselRelativeTranslation);

  // If the mother's doesn't intersects the dosel
  if(motherSV->GetCubicVolume()==0.) {
    GateMessage("Actor", 2, "[GateVoxelizedMass::VoxelIteration] " <<  motherPV->GetName() << " is not contained inside the dosel n°" << index << Gateendl);
    return std::make_pair(0.,0.);
  }

  GateMessage("Actor", 2, "[GateVoxelizedMass::VoxelIteration] "<< motherPV->GetName() <<" (after overlap mother-dosel)  : " << G4BestUnit(motherSV->GetCubicVolume(),"Volume") << Gateendl);

  if (IsLVParameterized(motherLV)) {
    GateMessage("Actor", 2, "[GateVoxelizedMass::VoxelIteration] WARNING: " << motherPV->GetName() << " is parameterized !" << Gateendl
        << " ==> Returning null mass ! " << Gateendl);
    return std::make_pair(0.,motherSV->GetCubicVolume());
  }

  // Calculation for daughter(s) ///////////////////////////////////////////
  if(motherLV->GetNoDaughters()>0)
  {
    for(int i=0;i<motherLV->GetNoDaughters();i++)
    {
      G4VPhysicalVolume*  daughterPV(motherLV->GetDaughter(i));
      G4VSolid*           daughterSV(daughterPV->GetLogicalVolume()->GetSolid());

      // Mother->Daughter relative translation and rotation
      G4RotationMatrix motherDaughterRelativeRotation    = daughterPV->GetObjectRotationValue();
      G4ThreeVector    motherDaughterRelativeTranslation = daughterPV->GetObjectTranslation();

      // Mother->GMother relative translation and rotation
      G4RotationMatrix motherGMotherRelativeRotation    = motherPV->GetObjectRotationValue();
      G4ThreeVector    motherGMotherRelativeTranslation = motherPV->GetObjectTranslation();

      // Daughter absolute translation and rotation
      G4RotationMatrix daughterAbsoluteRotation    = motherDaughterRelativeRotation * motherAbsoluteRotation;
      G4ThreeVector    daughterAbsoluteTranslation = motherGMotherRelativeRotation  * motherDaughterRelativeTranslation + motherAbsoluteTranslation;

      // Dosel->Daughter relative translation and rotation
      G4RotationMatrix doselDaughterRelativeRotation    = daughterAbsoluteRotation * doselAbsoluteRotation;
      G4ThreeVector    doselDaughterRelativeTranslation = doselAbsoluteTranslation - daughterAbsoluteTranslation;

      double motherCubicVolumeBefore(motherSV->GetCubicVolume());

      GateMessage("Actor", 2, "[GateVoxelizedMass::VoxelIteration] "<<motherPV->GetName()<<" (before sub. mother-daughter) : " << G4BestUnit(motherSV->GetCubicVolume(),"Volume") << " (daughterPV : "<< daughterPV->GetName() <<")"<< Gateendl);

      if(motherCubicVolumeBefore>0.)
      {
        // Substraction Dosel-Daughter
        motherSV=new G4SubtractionSolid(motherSV->GetName(),
                                        motherSV,
                                        daughterSV,
                                        &doselDaughterRelativeRotation,
                                        doselDaughterRelativeTranslation);

        GateMessage("Actor", 2, "[GateVoxelizedMass::VoxelIteration] "<<motherPV->GetName()<<" (after sub. mother-daughter)  : " << G4BestUnit(motherSV->GetCubicVolume(),"Volume") << " (daughterPV : "<<daughterPV->GetName()<<")"<< Gateendl);

        double diff((motherSV->GetCubicVolume()-motherCubicVolumeBefore)*100/motherCubicVolumeBefore);
        double substractionError(0.1);

        if(diff>substractionError)
          GateError("Error: CubicVolume after substraction is bigger than before !" << Gateendl << " difference: " << diff << "%" << Gateendl);
        else if(diff<=-substractionError)
        {
          std::pair<double,double> daughterIteration(VoxelIteration(daughterPV,Generation+1,daughterAbsoluteRotation,daughterAbsoluteTranslation,index));

          if (daughterIteration.first == 0.)
            GateMessage("Actor", 2, "[GateVoxelizedMass::VoxelIteration] WARNING: daughterIteration.first (mass) is null ! (daughterPhysicalVolume: " << daughterPV->GetName() << ")" << Gateendl
                << "Maybe " << daughterPV->GetName() << " is (partially) outside " << motherPV->GetName() << "." << Gateendl);
          if (daughterIteration.second == 0.)
            GateError("Error: daughterIteration.second (volume) is null ! (daughterPhysicalVolume: " << daughterPV->GetName() << ") "<< Gateendl
                << "Maybe " << daughterPV->GetName() << " is (partially) outside " << motherPV->GetName() << "." << Gateendl);

          motherProgenyMass+=daughterIteration.first;
          motherProgenyCubicVolume+=daughterIteration.second;
        }
      }
    }
  }
  //////////////////////////////////////////////////////////////////////////

  // Mother mass & volume //////////////////////////////////////////////////
  double motherCubicVolume(motherSV->GetCubicVolume());
  double motherDensity(motherLV->GetMaterial()->GetDensity());

  GateMessage("Actor", 2, "[GateVoxelizedMass::VoxelIteration] "<<motherPV->GetName()<<" density :" << G4BestUnit(motherDensity,"Volumic Mass")<< Gateendl);

  motherMass=motherCubicVolume*motherDensity;
  motherProgenyMass+=motherMass;
  motherProgenyCubicVolume+=motherCubicVolume;
  //////////////////////////////////////////////////////////////////////////

  // Saving ////////////////////////////////////////////////////////////////
  mCubicVolume[index].push_back(std::make_pair(motherSV->GetName(),motherCubicVolume));
  mMass[index].push_back       (std::make_pair(motherSV->GetName(),motherMass));
  //////////////////////////////////////////////////////////////////////////

  if (mHasFilter && mVolumeFilter != "" && motherSV->GetName() == mVolumeFilter) {
    GateMessage("Actor", 2, "[GateVoxelizedMass::VoxelIteration] VolumeFilter: Filtered volume " <<  motherSV->GetName() << " finded ! (Generation n°" << Generation << ")" << Gateendl);
    mFilteredVolumeMass        = motherMass;
    mFilteredVolumeCubicVolume = motherCubicVolume;
    mIsFilteredVolumeProcessed = true;
  }

  if(motherProgenyMass==0.)
    GateError("Error: motherProgenyMass is null ! (motherPhysicalVolume : "<<motherPV->GetName()<<")"<<Gateendl);
  if(motherProgenyCubicVolume==0.)
    GateError("Error: motherProgenyCubicVolume is null ! (motherPhysicalVolume : "<<motherPV->GetName()<<")"<<Gateendl);

  if (Generation==0)
  {
    double diff((motherProgenyCubicVolume-doselSV->GetCubicVolume())*100/doselSV->GetCubicVolume());
    double substractionError(0.5);

    if(std::abs(diff)>substractionError)
      GateError("Error: Dosel n°" << index << " is wrongly reconstructed !" << Gateendl <<
                "                            dosel (theorical)     = " << G4BestUnit(doselSV->GetCubicVolume(),"Volume") << Gateendl <<
                "                            dosel (reconstructed) = " << G4BestUnit(motherProgenyCubicVolume,"Volume") << Gateendl <<
                "                            difference            = " << diff << "%" << Gateendl );

    GateMessage("Actor", 2, "[GateVoxelizedMass::VoxelIteration] Dosel n°"<< index << " informations :" << Gateendl <<
                            "                                     motherProgenyMass        = " << G4BestUnit(motherProgenyMass,"Mass") << Gateendl <<
                            "                                     motherProgenyCubicVolume = " << G4BestUnit(motherProgenyCubicVolume,"Volume") << Gateendl);

    if (mHasFilter && mVolumeFilter != "") {
      if (mIsFilteredVolumeProcessed)
        return std::make_pair(mFilteredVolumeMass,mFilteredVolumeCubicVolume);
      else
        return std::make_pair(0.,0.);
    }
  }

  return std::make_pair(motherProgenyMass,motherProgenyCubicVolume);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateVoxelizedMass::GetPartialVolumeWithSV(const int index,const G4String SVName)
{
  if(mIsParameterised)
    GateError("Error: GateVoxelizedMass::GetPartialVolumeWithSV: This method doesn't work with voxelized volumes !"<<Gateendl);

  if(mCubicVolume[index].empty())
    GetVoxelMass(index);

  for(size_t i=0;i<mCubicVolume[index].size();i++)
    if(mCubicVolume[index][i].first==SVName)
      return mCubicVolume[index][i].second;

  GateError("!!! ERROR : GateVoxelizedMass::GetPartialVolume : Can't find "<<SVName<<" inside the dosel n°"<<index<<" !"<<Gateendl);
  return -1.;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateVoxelizedMass::GetPartialMassWithSV(const int index,const G4String SVName)
{
  if(mIsParameterised)
    GateError("Error: GateVoxelizedMass::GetPartialMassWithSV: This method doesn't work with voxelized volumes !"<<Gateendl);

  if(mMass[index].empty())
    GetVoxelMass(index);

  for(size_t i=0;i<mMass[index].size();i++)
    if(mMass[index][i].first==SVName)
      return mMass[index][i].second;

  GateError("!!! ERROR : GateVoxelizedMass::GetPartialMass : Can't find "<<SVName<<" inside the dosel n°"<<index<<" !"<<Gateendl);
  return -1.;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateVoxelizedMass::GetPartialVolumeWithMatName(const int index)
{
  if(mMaterialFilter=="")
    GateError("Error: GateVoxelizedMass::GetPartialVolumeWithMatName: No material filter defined !"<<Gateendl);

  if(!mIsParameterised)
    GateError("Error: GateVoxelizedMass::GetPartialVolumeWithMatName: This method only work with voxelized volumes !"<<Gateendl);

  if(!mIsVecGenerated)
    GenerateVectors();

  return doselReconstructedCubicVolume[index];
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateVoxelizedMass::GetPartialMassWithMatName(const int index)
{
  return GetVoxelMass(index);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
int GateVoxelizedMass::GetNumberOfVolumes(const int index)
{
 return mMass[index].size();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateVoxelizedMass::GetTotalVolume()
{
  return mImage.GetVoxelVolume();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVoxelizedMass::SetEdep(const int index,const G4String SVName,const double edep)
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
double GateVoxelizedMass::GetMaxDose(const int index)
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
GateImageDouble GateVoxelizedMass::UpdateImage(GateImageDouble image)
{
  if(mIsInitialized)
  {
    std::vector<double> vector;
    vector.clear();
    vector=GetVoxelMassVector();

    for(size_t i=0;i<vector.size();i++)
      image.AddValue(i,vector[i]/kg);
  }
  return image;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVoxelizedMass::SetMaterialFilter(const G4String MatName)
{
  if (MatName != "") {
    if (mHasFilter)
      GateError( "Error: GateVoxelizedMass::SetMaterialFilter: material filter is not compatible with other filters !" << Gateendl);
    else if (mHasExternalMassImage)
      GateError( "Error: GateVoxelizedMass::SetMaterialFilter: mass image importation is not compatible with filters !" << Gateendl);
    else {
      mMaterialFilter=MatName;
      mHasFilter=true;
    }
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVoxelizedMass::SetVolumeFilter(const G4String VolName)
{
  if (VolName != "") {
    if (mHasFilter)
      GateError( "Error: GateVoxelizedMass::SetVolumeFilter: volume filter is not compatible with other filters !" << Gateendl);
    else if (mHasExternalMassImage)
      GateError( "Error: GateVoxelizedMass::SetVolumeFilter: mass image importation is not compatible with filters !" << Gateendl);
    else {
      mVolumeFilter=VolName+"_solid";
      mHasFilter=true;
    }
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVoxelizedMass::SetExternalMassImage(const G4String extMassFile)
{
  if(extMassFile != "")
  {
    mMassFile=extMassFile;
    mHasExternalMassImage=true;

    if(mHasFilter)
      GateError( "Error: GateVoxelizedMass::SetExternalMassImage: mass image importation is not compatible with filters !" << Gateendl);
  }
}
//-----------------------------------------------------------------------------
