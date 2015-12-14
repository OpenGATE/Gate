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

#include <G4IntersectionSolid.hh>
#include <G4SubtractionSolid.hh>
#include <G4PhysicalVolumeStore.hh>
#include <G4VSolid.hh>
#include <G4Box.hh>
#include <G4VPVParameterisation.hh>

#include <ctime>

//-----------------------------------------------------------------------------
void GateVoxelizedMass::Initialize(const G4String mExtVolumeName, const GateImageDouble mExtImage,const G4String mExtMassFile)
{
  mMassFile=mExtMassFile;
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
  if(mMassFile!="")
  {
    mMassImage.Read(mMassFile);
    if(//mMassImage.GetHalfSize()!=mImage.GetHalfSize()||
       //mMassImage.GetResolution()!=mImage.GetResolution()||
       mMassImage.GetNumberOfValues()!=mImage.GetNumberOfValues())
    {
      /*G4cout<<"mMassImage.GetHalfSize()="<<mMassImage.GetHalfSize()<<G4endl
            <<"mImage.GetHalfSize()="<<mImage.GetHalfSize()<<G4endl;
      G4cout<<"mMassImage.GetResolution()="<<mMassImage.GetResolution()<<G4endl
            <<"mImage.GetResolution()="<<mImage.GetResolution()<<G4endl;
      G4cout<<"mMassImage.GetNumberOfValues()="<<mMassImage.GetNumberOfValues()<<G4endl
            <<"mImage.GetNumberOfValues()="<<mImage.GetNumberOfValues()<<G4endl;
      GateError("!!! ERROR : "<<mMassFile<<" hasn't the right size and resolution."<<Gateendl);*/
      GateError("!!! ERROR : "<<mMassFile<<" hasn't the right number of values."<<Gateendl);
    }
    doselExternalMass.resize(mMassImage.GetNumberOfValues(),-1.);
    for(int i=0;i<mMassImage.GetNumberOfValues();i++)
    {
      doselExternalMass[i]=mMassImage.GetValue(i)*kg;
      //G4cout<<"TEST : "<<mMassImage.GetValue(i)<<G4endl;
    }
  }

  DAPV=G4PhysicalVolumeStore::GetInstance()->GetVolume(mVolumeName+"_phys");
  DALV=DAPV->GetLogicalVolume();

  vectorSV.clear();
  doselReconstructedMass.clear();
  doselReconstructedCubicVolume.clear();
  doselReconstructedMass.resize(mImage.GetNumberOfValues(),-1.);
  doselReconstructedCubicVolume.resize(mImage.GetNumberOfValues(),-1.);

  doselSV=new G4Box("DoselSV",
                    mImage.GetVoxelSize().getX()/2.0,
                    mImage.GetVoxelSize().getY()/2.0,
                    mImage.GetVoxelSize().getZ()/2.0);

  if(DALV->GetNoDaughters()==1&&DALV->GetDaughter(0)->IsParameterised())
  {
    mIsParameterised=true;
    if(doselExternalMass.size()==0)
      GateVoxelizedMass::GenerateVoxels();
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateVoxelizedMass::GetVoxelMass(const int index)
{
  if(doselExternalMass.size()>0)
    return doselExternalMass[index];

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

  G4cout<<G4endl<<"================================================================"<<G4endl;
  G4cout<<" * Total voxelized mass calculation in progress, please wait ... "<<G4endl;

  doselReconstructedTotalCubicVolume=0.;
  doselReconstructedTotalMass=0.;

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

    if(difftime(timer4,timer1)>=60&&i%100==0)
    {
      std::cout<<" "<<i*100/mImage.GetNumberOfValues()<<"%\r"<<std::flush;
      // Experimental
      /*seconds=(mImage.GetNumberOfValues()-i)*difftime(timer4,timer3);
      if(seconds!=0.) std::cout<<"Estimated remaining time : "<<seconds/60<<"min"<<seconds%60<<"s ("<<seconds<<"s)                \r"<<std::flush;*/
    }
  }

  time(&timer2);
  seconds=difftime(timer2,timer1);

  G4cout<<" * SUMMARY : Mass calculation for voxelized volume :"<<G4endl;
  G4cout<<"     Time elapsed : "<<seconds/60<<"min"<<seconds%60<<"s ("<<seconds<<"s)"<<G4endl;
  if(mIsParameterised)
    G4cout<<"     Number of voxels : "<<DALV->GetDaughter(0)->GetMultiplicity()<<G4endl;
  G4cout<<"     Number of dosels : "<<mImage.GetNumberOfValues()<<G4endl;
  G4cout<<"     Dosels reconstructed total mass : "<<G4BestUnit(doselReconstructedTotalMass,"Mass")<<G4endl;
  G4cout<<"     Dosels reconstructed total cubic volume : "<<G4BestUnit(doselReconstructedTotalCubicVolume,"Volume")<<G4endl;
  G4cout<<"================================================================"<<G4endl<<G4endl;

  mIsVecGenerated=true;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVoxelizedMass::GenerateVoxels()
{
  G4VPhysicalVolume*     daughterPV(DALV->GetDaughter(0));
  G4VPVParameterisation* daughterParameterisation(daughterPV->GetParameterisation());

  DABox=(G4Box*)DALV->GetSolid();
  voxelBox=(G4Box*)daughterParameterisation->ComputeSolid(0,daughterPV);

  const int nxVoxel=round(DABox->GetXHalfLength()/voxelBox->GetXHalfLength()),
            nyVoxel=round(DABox->GetYHalfLength()/voxelBox->GetYHalfLength()),
            nzVoxel=round(DABox->GetZHalfLength()/voxelBox->GetZHalfLength());

  const int nxDosel=round(DABox->GetXHalfLength()/(mImage.GetVoxelSize().getX()/2.)),
            nyDosel=round(DABox->GetYHalfLength()/(mImage.GetVoxelSize().getY()/2.)),
            nzDosel=round(DABox->GetZHalfLength()/(mImage.GetVoxelSize().getZ()/2.));

  if(nxDosel>nxVoxel||nyDosel>nyVoxel||nzDosel>nzVoxel)
      GateError("!!! ERROR : The dosel resolution is smaller than the voxel resolution !!!"<<Gateendl);

  voxelCubicVolume.resize(nxVoxel);
  voxelMass.resize(nxVoxel);
  for(int x=0;x<nxVoxel;x++)
  {
    voxelCubicVolume[x].resize(nyVoxel);
    voxelMass[x].resize(nyVoxel);
    for(int y=0;y<nyVoxel;y++)
    {
      voxelCubicVolume[x][y].resize(nzVoxel,-1.);
      voxelMass[x][y].resize(nzVoxel,-1.);
    }
  }

  for(signed long int i=0;i<daughterPV->GetMultiplicity();i++)
  {
    daughterParameterisation->ComputeTransformation(i,daughterPV);

    const int xVoxel(round((DABox->GetXHalfLength()+daughterPV->GetTranslation().getX()-voxelBox->GetXHalfLength())/(voxelBox->GetXHalfLength()*2.0))),
              yVoxel(round((DABox->GetYHalfLength()+daughterPV->GetTranslation().getY()-voxelBox->GetYHalfLength())/(voxelBox->GetYHalfLength()*2.0))),
              zVoxel(round((DABox->GetZHalfLength()+daughterPV->GetTranslation().getZ()-voxelBox->GetZHalfLength())/(voxelBox->GetZHalfLength()*2.0)));

    if(xVoxel>=nxVoxel||yVoxel>=nyVoxel||zVoxel>=nzVoxel)
      GateError("!!! ERROR : Too many voxels !!! (xVoxel="<<xVoxel<<",yVoxel="<<yVoxel<<",zVoxel="<<zVoxel<<")"<<Gateendl);

    voxelCubicVolume[xVoxel][yVoxel][zVoxel]=daughterParameterisation->ComputeSolid(i,daughterPV)->GetCubicVolume();
    voxelMass[xVoxel][yVoxel][zVoxel]=daughterParameterisation->ComputeMaterial(i,daughterPV)->GetDensity()*daughterParameterisation->ComputeSolid(i,daughterPV)->GetCubicVolume();
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVoxelizedMass::GenerateDosels(const int index)
{
  // INFO : Dimension of the vectors : x = 0, y = 1, z = 2

  doselMin.resize(3,-1.);
  doselMin[0]=(DABox->GetXHalfLength()+mImage.GetVoxelCenterFromIndex(index).getX()-mImage.GetVoxelSize().getX()/2.0)/(voxelBox->GetXHalfLength()*2.0);
  doselMin[1]=(DABox->GetYHalfLength()+mImage.GetVoxelCenterFromIndex(index).getY()-mImage.GetVoxelSize().getY()/2.0)/(voxelBox->GetYHalfLength()*2.0);
  doselMin[2]=(DABox->GetZHalfLength()+mImage.GetVoxelCenterFromIndex(index).getZ()-mImage.GetVoxelSize().getZ()/2.0)/(voxelBox->GetZHalfLength()*2.0);

  doselMax.resize(3,-1.);
  doselMax[0]=(DABox->GetXHalfLength()+mImage.GetVoxelCenterFromIndex(index).getX()+mImage.GetVoxelSize().getX()/2.0)/(voxelBox->GetXHalfLength()*2.0);
  doselMax[1]=(DABox->GetYHalfLength()+mImage.GetVoxelCenterFromIndex(index).getY()+mImage.GetVoxelSize().getY()/2.0)/(voxelBox->GetYHalfLength()*2.0);
  doselMax[2]=(DABox->GetZHalfLength()+mImage.GetVoxelCenterFromIndex(index).getZ()+mImage.GetVoxelSize().getZ()/2.0)/(voxelBox->GetZHalfLength()*2.0);
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
              coef[dim].push_back(abs((1-fmod(doselMin[dim],1))-fmod(doselMax[dim],1))); //FIXME ?
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
            {
              double coefVox(coef[0][xVox]*coef[1][yVox]*coef[2][zVox]);

              doselReconstructedCubicVolume[index]+=voxelCubicVolume[coord[0][xVox]][coord[1][yVox]][coord[2][zVox]]*coefVox;
              doselReconstructedMass[index]+=voxelMass[coord[0][xVox]][coord[1][yVox]][coord[2][zVox]]*coefVox;
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
  //FIXME : Doesn't work with daughter overlapping its mother.

  if(motherPV->IsParameterised())
    GateError("The volume "<<motherPV->GetName()<<" is parameterized !"<<Gateendl<<"Please attach the Actor directly on this volume !"<<Gateendl);

  G4LogicalVolume* motherLV(motherPV->GetLogicalVolume());
  G4VSolid* motherSV(motherLV->GetSolid());
  double motherMass(0.);
  double motherDensity(motherLV->GetMaterial()->GetDensity());
  double motherProgenyMass(0.);
  double motherProgenyCubicVolume(0.);

  // Calculation of dosel's local rotation and translation
  G4RotationMatrix doselRotation(mImage.GetTransformMatrix());
  G4ThreeVector    doselTranslation(mImage.GetVoxelCenterFromIndex(index));

  if(Generation>0)
  {
    doselTranslation-=motherTranslation;
    doselRotation=doselRotation.transform(motherRotation);
  }

  // Overlap Mother-Dosel
  motherSV=new G4IntersectionSolid(motherSV->GetName(),//+"∩"+doselSV->GetName(),
                                  doselSV, 
                                  motherSV,
                                  &doselRotation, // Local rotation
                                  doselTranslation); // Local translation

  // If the mother's doesn't intersects the dosel
  if(motherSV->GetCubicVolume()==0.)
    return std::make_pair(0.,0.);

  // Calculation for daughter(s) ///////////////////////////////////////////
  if(motherLV->GetNoDaughters()>0) 
  {
    for(int i=0;i<motherLV->GetNoDaughters();i++)
    {
      G4VPhysicalVolume*  daughterPV(motherLV->GetDaughter(i));
      G4VSolid*           daughterSV(daughterPV->GetLogicalVolume()->GetSolid());

      // Relative translation and rotation.
      G4RotationMatrix daughterRotation(daughterPV->GetObjectRotationValue());
      G4ThreeVector    daughterTranslation(daughterPV->GetObjectTranslation());

      // Substraction Mother-Daughter
      motherSV=new G4SubtractionSolid(motherSV->GetName(),//+"-"+daughterSV->GetName(),
                                      motherSV, // Already overlapped with voxel volume
                                      daughterSV,
                                      &daughterRotation, // Local rotation
                                      daughterTranslation); // Local translation
    
      // Daughter absolute translation and rotation
      if(Generation>0)
      {
        daughterTranslation=motherTranslation+daughterPV->GetObjectTranslation().transform(motherRotation);
        daughterRotation=daughterRotation.transform(motherRotation);
      }

      std::pair<double,double> daughterIteration(VoxelIteration(daughterPV,Generation+1,daughterRotation,daughterTranslation,index));

      motherProgenyMass+=daughterIteration.first; 
      motherProgenyCubicVolume+=daughterIteration.second;
    }
  }
  //////////////////////////////////////////////////////////////////////////

  // Mother mass & volume //////////////////////////////////////////////////
  double motherCubicVolume(motherSV->GetCubicVolume());
  motherMass=motherCubicVolume*motherDensity;
  motherProgenyMass+=motherMass;
  motherProgenyCubicVolume+=motherCubicVolume;
  //////////////////////////////////////////////////////////////////////////

  // Saving ////////////////////////////////////////////////////////////////
  mCubicVolume[index].push_back(std::make_pair(motherSV->GetName(),motherCubicVolume));
  mMass[index].push_back       (std::make_pair(motherSV->GetName(),motherMass));
  //////////////////////////////////////////////////////////////////////////

  return std::make_pair(motherProgenyMass,motherProgenyCubicVolume);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
double GateVoxelizedMass::GetPartialVolume(const int index,const G4String SVName)
{
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
double GateVoxelizedMass::GetPartialMass(const int index,const G4String SVName)
{
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

  return edepmax/GetPartialMass(index,SVName);
}
//-----------------------------------------------------------------------------
