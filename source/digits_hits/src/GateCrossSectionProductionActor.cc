/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


/*
  \brief Class GateCrossSectionProductionActor :
  \brief
*/

#ifndef GATECROSSSECTIONPRODUCTIONACTOR_CC
#define GATECROSSSECTIONPRODUCTIONACTOR_CC

#include "GateCrossSectionProductionActor.hh"
#include "GateMiscFunctions.hh"
#include <sys/time.h>
//-----------------------------------------------------------------------------
GateCrossSectionProductionActor::GateCrossSectionProductionActor(G4String name, G4int depth):
  GateVImageActor(name,depth) {
  GateDebugMessageInc("Actor",4,"GateCrossSectionProductionActor() -- begin"<<G4endl);
  Na=6.02e+23;
  mCurrentEvent=0;
  nb_elemt_C12_in_table=-1;
  nb_elemt_O16_in_table=-1;
  mIsotopeFilename="prod_C11.hdr";
  //mIsotopeFilename="stat_C11"+G4String(getExtension(mSaveFilename));
  //mIsotopeFilename="energy_C11"+G4String(getExtension(mSaveFilename));
  pMessenger = new GateCrossSectionProductionActorMessenger(this);
  threshold_energy_O16 = 15.53;//MeV
  threshold_energy_C12 = 17.59;//MeV

  m_IsO15 = false;
  m_IsC11 = true;


  A_12 =12.0;
  A_16 =16.0;
  max_energy_cross_section = 0.;
  //Units : (MeV , mbarn)
  SectionTableC11_C12.insert(std::pair<float, float> (  17.59, 0. ) );
  SectionTableC11_C12.insert(std::pair<float, float> (  19.19,	13.56) );
  SectionTableC11_C12.insert(std::pair<float, float> ( 19.72,	24.66) );
  SectionTableC11_C12.insert(std::pair<float, float> (20.79,	34.93) );
  SectionTableC11_C12.insert(std::pair<float, float> (22.39,	43.56) );
  SectionTableC11_C12.insert(std::pair<float, float> (23.99,	53.42) );
  SectionTableC11_C12.insert(std::pair<float, float> (25.59,	63.29) );
  SectionTableC11_C12.insert(std::pair<float, float> (27.19,	76.44) );
  SectionTableC11_C12.insert(std::pair<float, float> (28.78,	78.08) );
  SectionTableC11_C12.insert(std::pair<float, float> (29.32,	83.01) );
  SectionTableC11_C12.insert(std::pair<float, float> (31.98,	86.71) );
  SectionTableC11_C12.insert(std::pair<float, float> (32.52,	88.77) );
  SectionTableC11_C12.insert(std::pair<float, float> (38.91,	87.12) );
  SectionTableC11_C12.insert(std::pair<float, float> (49.57,	87.12) );
  SectionTableC11_C12.insert(std::pair<float, float> (56.5,	82.6) );
  SectionTableC11_C12.insert(std::pair<float, float> (65.03,	78.08) );
  SectionTableC11_C12.insert(std::pair<float, float> (73.03,	73.97) );
  SectionTableC11_C12.insert(std::pair<float, float> (81.02,	69.45) );
  SectionTableC11_C12.insert(std::pair<float, float> (85.82,	67.4) );
  SectionTableC11_C12.insert(std::pair<float, float> (90.09,	64.93) );
  SectionTableC11_C12.insert(std::pair<float, float> (96.48,	62.88) );
  SectionTableC11_C12.insert(std::pair<float, float> (102.9,	60.41) );
  SectionTableC11_C12.insert(std::pair<float, float> (110.3,	56.71) );
  SectionTableC11_C12.insert(std::pair<float, float> (118.3,	53.42) );
  SectionTableC11_C12.insert(std::pair<float, float> (127.9,	50.55) );
  SectionTableC11_C12.insert(std::pair<float, float> (138.1,	48.08) );
  SectionTableC11_C12.insert(std::pair<float, float> (148.7,	45.21) );
  SectionTableC11_C12.insert(std::pair<float, float> (158.3,	43.97) );
  SectionTableC11_C12.insert(std::pair<float, float> (171.1,	42.33) );
  SectionTableC11_C12.insert(std::pair<float, float> (185.,	40.68) );
  SectionTableC11_C12.insert(std::pair<float, float> (196.2,	39.04) );
  SectionTableC11_C12.insert(std::pair<float, float> (208.4,	38.63) );
  SectionTableC11_C12.insert(std::pair<float, float> (219.6,	37.81) );
  SectionTableC11_C12.insert(std::pair<float, float> (232.4,	37.4) );
  SectionTableC11_C12.insert(std::pair<float, float> (246.3,	37.81) );
  SectionTableC11_C12.insert(std::pair<float, float> (249.5,	37.81) );

  SectionTableC11_O16.insert(std::pair<float, float> (29.63,	0.) );
  SectionTableC11_O16.insert(std::pair<float, float> (31.19,	1.978) );
  SectionTableC11_O16.insert(std::pair<float, float> (32.74	,4.121) );
  SectionTableC11_O16.insert(std::pair<float, float> (34.82,	6.758) );
  SectionTableC11_O16.insert(std::pair<float, float> (36.38,	9.973) );
  SectionTableC11_O16.insert(std::pair<float, float> (38.98,	13.27) );
  SectionTableC11_O16.insert(std::pair<float, float> (40.02,	15.58) );
  SectionTableC11_O16.insert(std::pair<float, float> (42.1	,17.47) );
  SectionTableC11_O16.insert(std::pair<float, float> (44.18	,19.45) );
  SectionTableC11_O16.insert(std::pair<float, float> (45.74	,21.43) );
  SectionTableC11_O16.insert(std::pair<float, float> (48.34	,23.49) );
  SectionTableC11_O16.insert(std::pair<float, float> (49.9	,25.38) );
  SectionTableC11_O16.insert(std::pair<float, float> (59.25	,26.87) );
  SectionTableC11_O16.insert(std::pair<float, float> (64.97	,25.47) );
  SectionTableC11_O16.insert(std::pair<float, float> (71.21	,24.07) );
  SectionTableC11_O16.insert(std::pair<float, float> (77.96	,22.42) );
  SectionTableC11_O16.insert(std::pair<float, float> (82.64	,21.51) );
  SectionTableC11_O16.insert(std::pair<float, float> (89.4	,20.69) );
  SectionTableC11_O16.insert(std::pair<float, float> (96.67	,20.11) );
  SectionTableC11_O16.insert(std::pair<float, float> (102.9	,19.86) );
  SectionTableC11_O16.insert(std::pair<float, float> (109.1	,19.86) );
  SectionTableC11_O16.insert(std::pair<float, float> (114.3	,19.53) );
  SectionTableC11_O16.insert(std::pair<float, float> (120.1	,19.29) );
  SectionTableC11_O16.insert(std::pair<float, float> (128.9	,19.45) );
  SectionTableC11_O16.insert(std::pair<float, float> (139.3	,18.87) );
  SectionTableC11_O16.insert(std::pair<float, float> (147.6	,18.96) );
  SectionTableC11_O16.insert(std::pair<float, float> (155.9	,19.04) );
  SectionTableC11_O16.insert(std::pair<float, float> (166.8	,19.2) );
  SectionTableC11_O16.insert(std::pair<float, float> (173.6	,19.37) );
  SectionTableC11_O16.insert(std::pair<float, float> (178.8	,19.53) );
  SectionTableC11_O16.insert(std::pair<float, float> (187.6	,19.37) );
  SectionTableC11_O16.insert(std::pair<float, float> (198.5	,19.04) );
  SectionTableC11_O16.insert(std::pair<float, float> (208.4	,19.29) );
  SectionTableC11_O16.insert(std::pair<float, float> (218.8	,19.62) );
  SectionTableC11_O16.insert(std::pair<float, float> (228.7	,19.62) );
  SectionTableC11_O16.insert(std::pair<float, float> (237.5	,19.7) );
  SectionTableC11_O16.insert(std::pair<float, float> (250.	,19.86) );


  SectionTableO15_O16.insert(std::pair<float, float> (15.53,	0.) );
  SectionTableO15_O16.insert(std::pair<float, float> (16.5	,6.733) );
  SectionTableO15_O16.insert(std::pair<float, float> (18.45,	14.21) );
  SectionTableO15_O16.insert(std::pair<float, float> (19.42,	21.45) );
  SectionTableO15_O16.insert(std::pair<float, float> (20.87,	31.67) );
  SectionTableO15_O16.insert(std::pair<float, float> (22.33,	43.14) );
  SectionTableO15_O16.insert(std::pair<float, float> (23.3,	54.36) );
  SectionTableO15_O16.insert(std::pair<float, float> (24.27,	62.84) );
  SectionTableO15_O16.insert(std::pair<float, float> (25.24,	71.07) );
  SectionTableO15_O16.insert(std::pair<float, float> (26.7,	79.55) );
  SectionTableO15_O16.insert(std::pair<float, float> (27.18,	85.29) );
  SectionTableO15_O16.insert(std::pair<float, float> (28.64,	90.77) );
  SectionTableO15_O16.insert(std::pair<float, float> (29.13,	93.52) );
  SectionTableO15_O16.insert(std::pair<float, float> (30.58,	96.26) );
  SectionTableO15_O16.insert(std::pair<float, float> (34.47,	97.01) );
  SectionTableO15_O16.insert(std::pair<float, float> (37.38,	95.01) );
  SectionTableO15_O16.insert(std::pair<float, float> (39.32,	91.02) );
  SectionTableO15_O16.insert(std::pair<float, float> (42.23,	86.03) );
  SectionTableO15_O16.insert(std::pair<float, float> (45.63,	79.55) );
  SectionTableO15_O16.insert(std::pair<float, float> (48.06,	75.56) );
  SectionTableO15_O16.insert(std::pair<float, float> (51.94,	73.32) );
  SectionTableO15_O16.insert(std::pair<float, float> (57.28,	72.32) );
  SectionTableO15_O16.insert(std::pair<float, float> (61.65,	72.07) );
  SectionTableO15_O16.insert(std::pair<float, float> (66.99,	72.57) );
  SectionTableO15_O16.insert(std::pair<float, float> (70.87,	72.82) );
  SectionTableO15_O16.insert(std::pair<float, float> (72.82,	70.57) );
  SectionTableO15_O16.insert(std::pair<float, float> (74.76,	67.83) );
  SectionTableO15_O16.insert(std::pair<float, float> (77.18,	65.59) );
  SectionTableO15_O16.insert(std::pair<float, float> (80.1,	64.59) );
  SectionTableO15_O16.insert(std::pair<float, float> (85.44,	65.09) );
  SectionTableO15_O16.insert(std::pair<float, float> (91.26,	65.09) );
  SectionTableO15_O16.insert(std::pair<float, float> (96.6	,63.84) );
  SectionTableO15_O16.insert(std::pair<float, float> (101.9,	62.59) );
  SectionTableO15_O16.insert(std::pair<float, float> (106.8,	61.1) );
  SectionTableO15_O16.insert(std::pair<float, float> (111.7,	59.1) );
  SectionTableO15_O16.insert(std::pair<float, float> (115.5,	56.36) );
  SectionTableO15_O16.insert(std::pair<float, float> (119.4,	53.12) );
  SectionTableO15_O16.insert(std::pair<float, float> (124.8,	50.12) );
  SectionTableO15_O16.insert(std::pair<float, float> (129.6,	47.63) );
  SectionTableO15_O16.insert(std::pair<float, float> (134.,	45.39) );
  SectionTableO15_O16.insert(std::pair<float, float> (138.3, 	42.64) );
  SectionTableO15_O16.insert(std::pair<float, float> (144.2,	40.15) );
  SectionTableO15_O16.insert(std::pair<float, float> (150.,	39.15) );
  SectionTableO15_O16.insert(std::pair<float, float> (157.3,	37.91) );
  SectionTableO15_O16.insert(std::pair<float, float> (165.5,	36.91) );
  SectionTableO15_O16.insert(std::pair<float, float> (171.8,	36.41) );
  SectionTableO15_O16.insert(std::pair<float, float> (178.6,	35.66) );
  SectionTableO15_O16.insert(std::pair<float, float> (185.9,	35.16) );
  SectionTableO15_O16.insert(std::pair<float, float> (194.7,	34.91) );
  SectionTableO15_O16.insert(std::pair<float, float> (199.5,	34.91) );


  GateDebugMessageDec("Actor",4,"GateCrossSectionProductionActor() -- end"<<G4endl);


}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateCrossSectionProductionActor::~GateCrossSectionProductionActor()  {
  delete pMessenger;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateCrossSectionProductionActor::Construct() {
  GateDebugMessageInc("Actor", 4, "GateCrossSectionProductionActor -- Construct - begin" << G4endl);
  GateVImageActor::Construct();

  // Enable callbacks
  G4cout << "GateCrossSectionProductionActor::Construct" << G4endl;
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnablePostUserTrackingAction(true);
  EnableUserSteppingAction(true);
  EnableEndOfEventAction(true);
  if(m_IsC11){
    mIsotopeFilename = G4String(removeExtension(mSaveFilename))+"-C11."+G4String(getExtension(mSaveFilename));
    mIsotopeImage = new GateImageWithStatistic();
    SetOriginTransformAndFlagToImage(*mIsotopeImage);
    mIsotopeImage->EnableSquaredImage(false);
    mIsotopeImage->EnableUncertaintyImage(false);
    mIsotopeImage->SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mIsotopeImage->Allocate();
    mIsotopeImage->SetFilename(mIsotopeFilename);
  }


  SetOriginTransformAndFlagToImage(mEnergyImage);
  SetOriginTransformAndFlagToImage(mStatImage);
  SetOriginTransformAndFlagToImage(mDensityImage);
  SetOriginTransformAndFlagToImage(mfractionC12Image);
  SetOriginTransformAndFlagToImage(mfractionO16Image);
  SetOriginTransformAndFlagToImage(mEnergyImage_secondary);
  SetOriginTransformAndFlagToImage(mStatImage_secondary);

  mEnergyImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mEnergyImage.Allocate();

  mStatImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mStatImage.Allocate();
  //for secondaries particle
  mEnergyImage_secondary.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mEnergyImage_secondary.Allocate();

  mStatImage_secondary.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mStatImage_secondary.Allocate();

  mDensityImage.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mfractionC12Image.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
  mfractionO16Image.SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);


  mDensityImage.Allocate();
  mfractionC12Image.Allocate();
  mfractionO16Image.Allocate();

  G4cout << " in GateCrossSectionProductionActor::Construct " << m_IsO15 << G4endl;
  //pour O15
  if(m_IsO15){
    mIsotopeImage_O15 = new GateImageWithStatistic();
    SetOriginTransformAndFlagToImage(*mIsotopeImage_O15);
    mIsotopeFilename = G4String(removeExtension(mSaveFilename))+"-O15."+G4String(getExtension(mSaveFilename));
    mIsotopeImage_O15->SetResolutionAndHalfSize(mResolution, mHalfSize, mPosition);
    mIsotopeImage_O15->EnableSquaredImage(false);
    mIsotopeImage_O15->EnableUncertaintyImage(false);
    mIsotopeImage_O15->Allocate();
    mIsotopeImage_O15->SetFilename(mIsotopeFilename);
  }

  //to find the last element of each tables
  std::map <float, float>::iterator it_O15 = SectionTableO15_O16.end();
  G4double max_energy_O15 =  (--it_O15)->first;
  std::map <float, float>::iterator it_C11_C12 = SectionTableC11_C12.end();
  G4double max_energy_C11_C12 =  (--it_C11_C12)->first;
  std::map <float, float>::iterator it_C11_O16 = SectionTableC11_O16.end();
  G4double max_energy_C11_O16 =  (--it_C11_O16)->first;


  //map to sort maximum energy
  std::map <float, float> Energy_max;


  if(m_IsC11){
    //the set value is not important
    Energy_max[max_energy_C11_C12]=0.;
    Energy_max[max_energy_C11_O16]=0.;
  }
  if(m_IsO15){
    //the set value is not important
    Energy_max[max_energy_O15]=0.;

  }
  max_energy_cross_section = Energy_max.begin()->first;

  G4cout << " in GateCrossSectionProductionActor::Construct max_energy_cross_section = " << max_energy_cross_section<< G4endl;
  ResetData();
  GateMessageDec("Actor", 4, "GateCrossSectionProductionActor -- Construct - end" << G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateCrossSectionProductionActor::SaveData() {
  // mStatImage.SaveData(mCurrentEvent+1);

  if(m_IsC11){
    mIsotopeImage->SaveData(mCurrentEvent+1,false);
    G4double total_C11 =0.;
    for(int i = 0 ;i <mIsotopeImage->GetValueImage().GetNumberOfValues() ; i++){
      total_C11 += mIsotopeImage->GetValue(i);
    }

    //G4cout << " total_C11 " << total_C11 << G4endl;
  }

  if(m_IsO15){
    //G4cout << "is saving O15" << G4endl;
    mIsotopeImage_O15->SaveData(mCurrentEvent+1,false);
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateCrossSectionProductionActor::ResetData() {

  if(m_IsC11){
    mIsotopeImage->Reset();
  }
  if(m_IsO15){
    mIsotopeImage_O15->Reset();
  }

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateCrossSectionProductionActor::BeginOfRunAction(const G4Run * r) {
  GateVActor::BeginOfRunAction(r);
  GateDebugMessage("Actor", 3, "GateCrossSectionProductionActor -- Begin of Run" << G4endl);

  gettimeofday(&mTimeOfLastSaveEvent, NULL);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback at each event
void GateCrossSectionProductionActor::BeginOfEventAction(const G4Event * e) {
  GateVActor::BeginOfEventAction(e);
  mCurrentEvent++;


  GateDebugMessage("Actor", 3, "GateCrossSectionProductionActor -- Begin of Event: "<<mCurrentEvent << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateCrossSectionProductionActor::UserSteppingActionInVoxel(const int index, const G4Step* step) {
  GateDebugMessageInc("Actor", 4, "GateCrossSectionProductionActor -- UserSteppingActionInVoxel - begin" << G4endl);

  //double edep = step->GetTotalEnergyDeposit();
  double energy=0.;
  double density=0.;

  int index_for_Gate_Image =index;
  if(step->GetTrack()->GetParticleDefinition()->GetParticleName()=="proton"){

    G4Track * aTrack= step->GetTrack();
    G4int i_find_nb_elemt =0;

    while((nb_elemt_C12_in_table==-1 || nb_elemt_O16_in_table==-1 )&& i_find_nb_elemt<(G4int)step->GetPreStepPoint()->GetMaterial()->GetNumberOfElements() ){
      //G4cout << step->GetPreStepPoint()->GetMaterial()->GetNumberOfElements() << G4endl;
      //G4cout << i_find_nb_elemt << G4endl;
      //G4cout << step->GetPreStepPoint()->GetMaterial()->GetName()<< G4endl;
      //G4cout << step->GetPreStepPoint()->GetMaterial()->GetElement(i_find_nb_elemt)->GetZ()<< G4endl;
      if(step->GetPreStepPoint()->GetMaterial()->GetElement(i_find_nb_elemt)->GetZ()==6){nb_elemt_C12_in_table=i_find_nb_elemt;}
      if(step->GetPreStepPoint()->GetMaterial()->GetElement(i_find_nb_elemt)->GetZ()==8){nb_elemt_O16_in_table=i_find_nb_elemt;}
      i_find_nb_elemt++;
    }




    if((nb_elemt_C12_in_table==-1 || nb_elemt_O16_in_table==-1 )==true){
      //G4cout<< " problem to find an index or there is no C12 nor O16" << G4endl;
    }
    if(newTrack){
      energy = step->GetPreStepPoint()->GetKineticEnergy()/MeV;
      if(energy>=threshold_energy_C12){
        if(aTrack->GetTrackID()==1){
          if(energy>max_energy_cross_section){
            GateError("The CrossSectionActor " << GetObjectName() << " does not have this energy in data, please lower the energy or add data, the current limit is : " << max_energy_cross_section << " MeV, current energy is " << energy);
          }
          mEnergyImage.AddValue(index, energy);
          PixelValuePerEvent.insert(std::pair<int,int>(index,0));
          mStatImage.AddValue(index, 1);
        }else{
          mEnergyImage_secondary.AddValue(index, energy);
          PixelValuePerEvent_secondary.insert(std::pair<int,int>(index,0));
          mStatImage_secondary.AddValue(index, 1);

        }
	//G4cout << " & new track is adding " << energy << " in vox " << index << G4endl;
      }

      newTrack=false;
    }else{
      energy = step->GetTrack()->GetKineticEnergy()/MeV;
      if(energy>=threshold_energy_C12){
        if(aTrack->GetTrackID()==1){
          mEnergyImage.AddValue(index, energy);
          PixelValuePerEvent.insert(std::pair<int,int>(index,0));
          mStatImage.AddValue(index, 1);
        }else{
          mEnergyImage_secondary.AddValue(index, energy);
          PixelValuePerEvent_secondary.insert(std::pair<int,int>(index,0));
          mStatImage_secondary.AddValue(index, 1);

        }
	//G4cout << "is adding " << energy << " in vox " << index << G4endl;

      }

    }

    density = step->GetPreStepPoint()->GetMaterial()->GetDensity()/(gram/cm3);

    mDensityImage.SetValue(index_for_Gate_Image,density );
    if(nb_elemt_C12_in_table!= -1){
      mfractionC12Image.SetValue(index_for_Gate_Image,step->GetPreStepPoint()->GetMaterial()->GetFractionVector()[nb_elemt_C12_in_table]);
    }
    if(nb_elemt_O16_in_table!= -1){
      mfractionO16Image.SetValue(index_for_Gate_Image,step->GetPreStepPoint()->GetMaterial()->GetFractionVector()[nb_elemt_O16_in_table]);
    }


  }

  GateDebugMessageDec("Actor", 4, "GateCrossSectionProductionActor -- UserSteppingActionInVoxel -- end" << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateCrossSectionProductionActor::EndOfEventAction(const G4Event* eve)
{
  GateDebugMessage("Actor", 3, "GateCrossSectionProductionActor -- End of Event" << G4endl);
  double volume_vox=mIsotopeImage->GetValueImage().GetVoxelVolume()*1e-3;//switch to mm3 to cm3

  G4double prod =0.;
  G4double beam_entrance_section=mIsotopeImage->GetValueImage().GetVoxelSize().getX()*mIsotopeImage->GetValueImage().GetVoxelSize().getY()*1.0e-2; //in cm2

  for(std::map<int,int>::iterator i =PixelValuePerEvent.begin() ; i!=PixelValuePerEvent.end(); ++i){
    //in this condition either the pixel has already been treater or no detection in the voxel
    if(mStatImage.GetValue(i->first)!=0 && mEnergyImage.GetValue(i->first)> threshold_energy_O16 ){ //checking that the energy is higher than the threshold to speed up
      G4int vox_id = i->first;
      G4int  stat_in_vox = mStatImage.GetValue(vox_id);
      G4double energy_in_vox = mEnergyImage.GetValue(vox_id);
      G4double mean_energy= energy_in_vox/((G4double)stat_in_vox);
      G4double density_in_vox = mDensityImage.GetValue(vox_id );

      G4double f_C12 = mfractionC12Image.GetValue(vox_id );
      G4double f_O16 = mfractionO16Image.GetValue(vox_id );

      if(m_IsC11 && mean_energy >=threshold_energy_C12 /*&& nb_elemt_C12_in_table !=1*/){
        prod= volume_vox*Na*density_in_vox*f_C12/A_12*GetSectionEfficace(mean_energy,SectionTableC11_C12)*1e-24*1e-3/beam_entrance_section;
        mIsotopeImage->AddValue(vox_id,prod);

      }
      if(m_IsC11 && mean_energy >=threshold_energy_C12 /*&& nb_elemt_O16_in_table !=1*/){
        prod=volume_vox*Na*density_in_vox*f_O16/A_16*GetSectionEfficace(mean_energy,SectionTableC11_O16)*1e-24*1e-3/beam_entrance_section;
        mIsotopeImage->AddValue(vox_id,prod);
      }
      if(m_IsO15 && mean_energy >=threshold_energy_O16 /*&& nb_elemt_O16_in_table !=1*/){
        prod=volume_vox*Na*density_in_vox*f_O16/A_16*GetSectionEfficace(mean_energy,SectionTableO15_O16)*1e-24*1e-3/beam_entrance_section;
        mIsotopeImage_O15->AddValue(vox_id,prod);
      }
      /*G4cout << mIsotopeImage.GetValueImage().GetCoordinatesFromIndex(vox_id).getX()<< " "<< mIsotopeImage.GetValueImage().GetCoordinatesFromIndex(vox_id).getY()<< " "<< mIsotopeImage.GetValueImage().GetCoordinatesFromIndex(vox_id).getZ()<< " ";
	G4cout << "vox_id : " << vox_id <<" f_C12 : "<< f_C12 <<  " mean_energy " << mean_energy <<  " section eff =  " << GetSectionEfficace(mean_energy,SectionTableC11_C12) << G4endl;
      */
      //reset des images a l'index donné
      mEnergyImage.SetValue(vox_id,0.);
      mStatImage.SetValue(vox_id,0.);
    }
  }

  PixelValuePerEvent.clear();
  for(std::map<int,int>::iterator i =PixelValuePerEvent_secondary.begin() ; i!=PixelValuePerEvent_secondary.end(); ++i){
    //in this condition either the pixel has already been treater or no detection in the voxel
    if(mStatImage_secondary.GetValue(i->first)!=0 && mEnergyImage_secondary.GetValue(i->first)> threshold_energy_O16 ){
      G4int vox_id = i->first;
      G4int  stat_in_vox = mStatImage_secondary.GetValue(vox_id);
      G4double energy_in_vox = mEnergyImage_secondary.GetValue(vox_id);
      G4double mean_energy= energy_in_vox/((G4double)stat_in_vox);
      G4double density_in_vox = mDensityImage.GetValue(vox_id );

      G4double f_C12 = mfractionC12Image.GetValue(vox_id );
      G4double f_O16 = mfractionO16Image.GetValue(vox_id );

      if(m_IsC11 && mean_energy >=threshold_energy_C12 /*&& nb_elemt_C12_in_table !=1*/){
        prod= volume_vox*Na*density_in_vox*f_C12/A_12*GetSectionEfficace(mean_energy,SectionTableC11_C12)*1e-24*1e-3/beam_entrance_section;
        mIsotopeImage->AddValue(vox_id,prod);
      }
      if(m_IsC11 && mean_energy >=threshold_energy_C12 /*&& nb_elemt_O16_in_table !=1*/){
        prod=volume_vox*Na*density_in_vox*f_O16/A_16*GetSectionEfficace(mean_energy,SectionTableC11_O16)*1e-24*1e-3/beam_entrance_section;
        mIsotopeImage->AddValue(vox_id,prod);
      }
      if(m_IsO15 && mean_energy >=threshold_energy_O16 /*&& nb_elemt_O16_in_table !=1*/){
        prod=volume_vox*Na*density_in_vox*f_O16/A_16*GetSectionEfficace(mean_energy,SectionTableO15_O16)*1e-24*1e-3/beam_entrance_section;
        mIsotopeImage_O15->AddValue(vox_id,prod);
      }

      /*G4cout << mIsotopeImage.GetValueImage().GetCoordinatesFromIndex(vox_id).getX()<< " "<< mIsotopeImage.GetValueImage().GetCoordinatesFromIndex(vox_id).getY()<< " "<< mIsotopeImage.GetValueImage().GetCoordinatesFromIndex(vox_id).getZ()<< " ";
	G4cout << "vox_id : " << vox_id <<" f_C12 : "<< f_C12 <<  " mean_energy " << mean_energy <<  " section eff =  " << GetSectionEfficace(mean_energy,SectionTableC11_C12) << G4endl;*/

      //reset images at a given index
      mEnergyImage_secondary.SetValue(vox_id,0.);
      mStatImage_secondary.SetValue(vox_id,0.);
    }
  }
  PixelValuePerEvent_secondary.clear();
  int ne =eve->GetEventID()+1;

  // Save every n events
  if ((ne != 0) && (mSaveEveryNEvents != 0))
    if (ne % mSaveEveryNEvents == 0){
      G4cout << "GateCrossSectionProductionActor::EndOfEventAction to Save " << G4endl;
      SaveData();
    }
  // Save every n seconds
  if (mSaveEveryNSeconds != 0) { // need to check time
    struct timeval end;
    gettimeofday(&end, NULL);
    long seconds  = end.tv_sec  - mTimeOfLastSaveEvent.tv_sec;
    if (seconds > mSaveEveryNSeconds) {
      //GateMessage("Core", 0, "Actor " << GetName() << " : " << mSaveEveryNSeconds << " seconds." << G4endl);
      SaveData();
      mTimeOfLastSaveEvent = end;
    }
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateCrossSectionProductionActor::PreUserTrackingAction(const GateVVolume *, const G4Track*)
{
  GateDebugMessage("Actor", 3, "GateCrossSectionProductionActor -- Begin of Track" << G4endl);

  newTrack = true; //nTrack++;

}

//-----------------------------------------------------------------------------
double GateCrossSectionProductionActor::GetSectionEfficace(double nrj,std::map<float, float>& MapSigma){

  float xa = -1. , ya,  xb=-1.0,  yb;

  std::map<float, float>::iterator found2 = MapSigma.begin(); //lower energy
  float dedx_sigma = found2->second;
  for (std::map<float, float>::iterator found = MapSigma.begin(); found != MapSigma.end(); ++found)
    {

      if(nrj < found2->first){

        return 0.0;

      }else{


        if (nrj < found->first ){
          xb =found->first ;
          yb =found->second ;
          if (found != MapSigma.begin()){

            found--;
            xa =found->first ;
            ya =found->second ;
          }else{
            return (found->second);
            //return 0.0;
          }
          //cout << InterpolLin (xa,ya,xb,yb, nrj) <<endl;
          return InterpolLin (xa,ya,xb,yb, nrj);
        }else if (nrj ==found->first )
          {
            //cout << " cas degalite " << endl;
            return (found->second);
          }

      }

    }
  return dedx_sigma;
}
float GateCrossSectionProductionActor::InterpolLin (float xa , float ya, float xb, float yb, float X)
{
  float Y ,a ,b;
  a= ( yb -ya)/(xb-xa);
  b = ya -a*xa;
  Y= a*X + b;
  return Y;
}
#endif
