/*----------------------
  GATE version name: gate_v...

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

//=======================================================
//	Created by Loïc Grevillot
//	General definition of the class
// This class allows for simulating treatment plans for Pencil Beam Scanning applications.
// The source need 2 inputs: a beam model of the system and a treatment plans
// It will simulate each single pencil beam of the treatment plan using the GateSourcePencilBeam class.
//=======================================================

//Modified by Hermann Fuchs
//Medical University Vienna
//Added missing Couch Angle
//Corrected angle calculation to be conform with DICOM standards
//Corrected energy distribution to conform to expected behaviour and publication

#ifndef GATESOURCETPSPENCILBEAM_CC
#define GATESOURCETPSPENCILBEAM_CC

#include "GateConfiguration.h"

#ifdef G4ANALYSIS_USE_ROOT
#include "GateSourceTPSPencilBeam.hh"
#include "G4Proton.hh"

//------------------------------------------------------------------------------------------------------
GateSourceTPSPencilBeam::GateSourceTPSPencilBeam(G4String name ):GateVSource( name ), mDistriGeneral(NULL)
{

  strcpy(mParticleType,"proton");
  mDistanceSMXToIsocenter=1000;
  mDistanceSMYToIsocenter=1000;
  mDistanceSourcePatient=500;
  pMessenger = new GateSourceTPSPencilBeamMessenger(this);
  mTestFlag=false;
  mCurrentParticleNumber=0;
  mCurrentSpot=0;
  mFlatGenerationFlag=false;
  mIsASourceDescriptionFile=false;
  mSpotIntensityAsNbProtons=false;
  mIsInitialized=false;
  mConvergentSource=false;
  mSelectedLayerID = -1; // all layer selected by default
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
GateSourceTPSPencilBeam::~GateSourceTPSPencilBeam()
{
  delete pMessenger;	// commented due to segfault
  //  for (int i=0; i<mPencilBeams.size(); i++)  { delete mPencilBeams[i]; }
  //FIXME segfault when uncommented
  //if (mDistriGeneral) delete mDistriGeneral;
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
void GateSourceTPSPencilBeam::GenerateVertex( G4Event* aEvent )
{

  if (!mIsInitialized){
    // get GATE random engine
    CLHEP::HepRandomEngine *engine = GateRandomEngine::GetInstance()->GetRandomEngine();

    //---------INITIALIZATION - START----------------------
    mIsInitialized=true;

    const int MAXLINE=256;
    char oneline[MAXLINE];
    int NbFields, FieldID, TotalMeterSet, NbOfLayers;
    double GantryAngle;
    double CouchAngle;
    double IsocenterPosition[3];
    double NbProtons;
    bool again=true;

    if (mIsASourceDescriptionFile) {
      LoadClinicalBeamProperties();
      GateMessage("Physic", 1, "[TPSPencilBeam] Source description file successfully loaded."<<G4endl);
    }
    else {cout<<"ERROR TPS PENCIL BEAM: No clinical beam loaded !\n"<<endl;
      exit (0);
    }

    ifstream inFile(mPlan);
    if (! inFile){
      G4cout << "ERROR TPS PENCIL BEAM:  cannot open Treatment plan file!\n"<<G4endl;
      exit (0);
    }

    // integrating the plan description file data
    while (inFile && again)
      {
        for (int i=0;i<9;i++) inFile.getline(oneline, MAXLINE);
        NbFields=atoi(oneline);
        for (int i=0;i<2*NbFields;i++) inFile.getline(oneline, MAXLINE);
        for (int i=0;i<2;i++) inFile.getline(oneline, MAXLINE);
        TotalMeterSet=atoi(oneline);

        for (int f=0; f<NbFields; f++){
          for (int i=0;i<4;i++) inFile.getline(oneline, MAXLINE);
          FieldID=atoi(oneline);

          for (int i=0;i<4;i++) inFile.getline(oneline, MAXLINE);
          GantryAngle=atof(oneline)*TMath::Pi()/180.;

          //MISSING COUCH ANGLE inserted
          for (int i=0;i<2;i++) inFile.getline(oneline, MAXLINE);
          CouchAngle=atof(oneline)*TMath::Pi()/180.;

          for (int i=0;i<2;i++) inFile.getline(oneline, MAXLINE);
          ReadLineTo3Doubles(IsocenterPosition, oneline);
          for (int i=0;i<2;i++) inFile.getline(oneline, MAXLINE);
          NbOfLayers=atoi(oneline);
          for (int i=0;i<2;i++) inFile.getline(oneline, MAXLINE);

          for (int j=0; j<NbOfLayers; j++) {

            for (int i=0;i<2;i++) inFile.getline(oneline, MAXLINE);
            int currentLayerID=atoi(oneline); // ControlPointIndex

            for (int i=0;i<6;i++) inFile.getline(oneline, MAXLINE);
            double energy=atof(oneline);

            for (int i=0;i<2;i++) inFile.getline(oneline, MAXLINE);
            int NbOfSpots=atof(oneline);
            for (int i=0;i<1;i++) inFile.getline(oneline, MAXLINE);
            if (mTestFlag){
              G4cout<<"TESTREAD NbFields "<<NbFields<<G4endl;
              G4cout<<"TESTREAD TotalMeterSet "<<TotalMeterSet<<G4endl;
              G4cout<<"TESTREAD FieldID "<<FieldID <<G4endl;
              G4cout<<"TESTREAD GantryAngle "<<GantryAngle <<G4endl;
              G4cout<<"TESTREAD CouchAngle "<<CouchAngle <<G4endl;
              G4cout<<"TESTREAD Layers N° "<<j<<G4endl;
              G4cout<<"TESTREAD NbOfSpots "<<NbOfSpots<<G4endl;
            }
            for(int k=0; k<NbOfSpots; k++){
              inFile.getline(oneline, MAXLINE);
              double SpotParameters[3];
              ReadLineTo3Doubles(SpotParameters, oneline);
              if (mTestFlag){
                G4cout<<"TESTREAD Spot N° "<<k<<"    parameters: "<<SpotParameters[0]<<" "<<SpotParameters[1]<<" "<<SpotParameters[2]<<G4endl;
              }

              //POSITION
              // To calculate the beam position with a gantry angle
              G4ThreeVector position;
              position[0]=SpotParameters[0]*(mDistanceSMXToIsocenter-mDistanceSourcePatient)/mDistanceSMXToIsocenter;
              position[1]=SpotParameters[1]*(mDistanceSMYToIsocenter-mDistanceSourcePatient)/mDistanceSMYToIsocenter;
              position[2]=mDistanceSourcePatient;
              //            position[0]=SpotParameters[0]*(mDistanceSMXToIsocenter-mDistanceSourcePatient)/mDistanceSMXToIsocenter;
              //	    position[1]=SpotParameters[1]*(mDistanceSMYToIsocenter-mDistanceSourcePatient)/mDistanceSMYToIsocenter;
              //	    position[2]=mDistanceSourcePatient;
              //correct orientation problem by rotation 90 degrees around x-Axis
              double xCorrection=90.*TMath::Pi()/180.;
              //            position.rotateX(xCorrection-CouchAngle);
              position.rotateX(xCorrection-CouchAngle);
              //if (GantryAngle!=0)//orig
              //            position.rotateY(GantryAngle);//orig
              //position.rotateY(GantryAngle);

              //include couch rotation
              //if (CouchAngle!=0)
              //            position.rotateY(CouchAngle);
              //include gantry rotation
              //if (GantryAngle!=0)
              position.rotateZ(GantryAngle);
              //            position.rotateX(-CouchAngle);

              if (mTestFlag){
                G4cout<<"TESTREAD Spot Effective source position "<<position[0]<<" "<<position[1]<<" "<<position[2]<<G4endl;
                G4cout<<"TESTREAD IsocenterPosition "<<IsocenterPosition[0]<<" "<<IsocenterPosition[1]<<" "<<IsocenterPosition[2]<<G4endl;
                G4cout<<"TESTREAD NbOfLayers "<<NbOfLayers<<G4endl;
              }

              //DIRECTION
              // To calculate the 3 required rotation angles to rotate the beam according to the direction set in the TPS
              G4ThreeVector rotation, direction, test;


              // GantryAngle at 0 (Default)
              //            //ORIGINAL BLOCK
              //            rotation[0]=TMath::Pi();
              //	    // deltaY in the patient plan
              //	    rotation[0]+=atan(SpotParameters[1]/mDistanceSMYToIsocenter);
              //	    // deltaX in the patient plan
              //	    rotation[1]=-atan(SpotParameters[0]/mDistanceSMXToIsocenter);
              //	    // no gantry head rotation
              //	    rotation[2]=0;
              //	    //set gantry angle rotation
              //            rotation[1]+=GantryAngle;


              //                        // GantryAngle at 0 (Default)
              //            //rotation[0]=TMath::Pi()+xCorrection;//270 degrees
              //            rotation[0]=-xCorrection;//270 degrees
              //
              //	    // deltaY in the patient plan
              //	    rotation[0]+=atan(SpotParameters[1]/mDistanceSMYToIsocenter);
              //	    // deltaX in the patient plan
              //	    rotation[2]=-atan(SpotParameters[0]/mDistanceSMXToIsocenter);
              //	    // no gantry head rotation
              //	    rotation[1]=0.;
              //	    //set gantry angle rotation
              //	    rotation[2]+=GantryAngle;//+CouchAngle;
              ////            rotation[2]+=CouchAngle;
              ////            rotation[0]+=-CouchAngle;//-couchAngle


              // GantryAngle at 0 (Default)
              //rotation[0]=TMath::Pi()+xCorrection;//270 degrees
              rotation[0]=-xCorrection;//270 degrees

              // deltaY in the patient plan
              double y=atan(SpotParameters[1]/mDistanceSMYToIsocenter);
              rotation[0]+=y;
              // deltaX in the patient plan
              double x=-atan(SpotParameters[0]/mDistanceSMXToIsocenter);
              double z=0.;
              rotation[1]=sin(CouchAngle)*(x) +cos(CouchAngle)*z;
              // no gantry head rotation
              rotation[2]=cos(CouchAngle)*(x) +sin(CouchAngle)*z;
              //set gantry angle rotation
              rotation[2]+=GantryAngle;//+CouchAngle;
              //            rotation[2]+=CouchAngle;
              rotation[0]+=-CouchAngle;//-couchAngle


              //            rotation[0]=0;//TMath::Pi();
              //	    // deltaY in the patient plan
              //	    rotation[0]+=atan(SpotParameters[1]/mDistanceSMYToIsocenter);
              //	    // deltaX in the patient plan
              //	    rotation[2]=-atan(SpotParameters[0]/mDistanceSMXToIsocenter);
              //	    // no gantry head rotation
              //	    rotation[1]=0;
              //	    //set gantry angle rotation
              //            //rotation[1]+=GantryAngle;
              //            rotation[1]-=GantryAngle;
              //            rotation[2]+=0.;

              //            G4cout<<"TESTREAD Spot Effective source position "<<position[0]<<" "<<position[1]<<" "<<position[2]<<G4endl;
              //            G4cout<<"TESTREAD source rotation "<<rotation[0]<<" "<<rotation[1]<<" "<<rotation[2]<<G4endl;
              //            G4cout<<"TESTREAD couch angle "<<CouchAngle<<G4endl;
              //            G4cout<<"TESTREAD gantry angle "<<GantryAngle<<G4endl;
              //            G4cout<<G4endl;
              if (mTestFlag){
                G4cout<<"TESTREAD source rotation "<<rotation[0]<<" "<<rotation[1]<<" "<<rotation[2]<<G4endl;
              }


              bool allowedField=true;
              for (unsigned int i=0; i<mNotAllowedFields.size(); i++) {
                if (FieldID==mNotAllowedFields[i]) allowedField=false;
              }

              bool allowedLayer=true;
              if ((mSelectedLayerID != -1) && (currentLayerID != mSelectedLayerID)) allowedLayer = false;

              if (allowedField && allowedLayer)  // loading the spots only for allowed fields
                {
                  GateSourcePencilBeam * Pencil = new GateSourcePencilBeam ("PencilBeam");
                  //Particle Type
                  Pencil->SetParticleType(mParticleType);
                  //Energy
                  Pencil->SetEnergy(GetEnergy(energy));
                  Pencil->SetSigmaEnergy(GetSigmaEnergy(energy));
                  //changed because obiously incorrect.
                  //Pencil->SetSigmaEnergy(GetSigmaEnergy(energy)*GetEnergy(energy)/100.);
                  //Weight

                  if (mSpotIntensityAsNbProtons) {
                    NbProtons=SpotParameters[2];
                  }
                  else {
                    NbProtons=ConvertMuToProtons(SpotParameters[2], GetEnergy(energy));
                  }

                  Pencil->SetWeight(NbProtons);
                  //G4cout<<"Nb of MU = "<<SpotParameters[2]<<", beam energy = "<<energy<<" MeV, corresponding to "<<NbProtons<<" protons."<<G4endl;
                  //Position
                  Pencil->SetPosition(position);
                  Pencil->SetSigmaX(GetSigmaX(energy));
                  Pencil->SetSigmaY(GetSigmaY(energy));
                  //Direction
                  Pencil->SetSigmaTheta(GetSigmaTheta(energy));
                  Pencil->SetEllipseXThetaArea(GetEllipseXThetaArea(energy));
                  Pencil->SetSigmaPhi(GetSigmaPhi(energy));
                  Pencil->SetEllipseYPhiArea(GetEllipseYPhiArea(energy));
                  Pencil->SetRotation(rotation);

                  //Correlation Position/Direction
                  if (mConvergentSource){
                    Pencil->SetEllipseXThetaRotationNorm("positive");	// convergent beam
                    Pencil->SetEllipseYPhiRotationNorm("positive");	// convergent beam
                  }
                  else{
                    Pencil->SetEllipseXThetaRotationNorm("negative");	// divergent beam
                    Pencil->SetEllipseYPhiRotationNorm("negative");	// divergent beam
                  }
                  Pencil->SetTestFlag(mTestFlag);
                  //new pencil added
                  mPencilBeams.push_back(Pencil);

                  if (mTestFlag){
                    cout<<"Energy\t"<<energy<<endl;
                    cout<<"SetEnergy\t"<<GetEnergy(energy)<<endl;
                    cout<<"SetSigmaEnergy\t"<<GetSigmaEnergy(energy)<<endl;
                    cout<<"SetSigmaX\t"<<GetSigmaX(energy)<<endl;
                    cout<<"SetSigmaY\t"<<GetSigmaY(energy)<<endl;
                    cout<<"SetSigmaTheta\t"<<GetSigmaTheta(energy)<<endl;
                    cout<<"SetSigmaPhi\t"<<GetSigmaPhi(energy)<<endl;
                    cout<<"SetEllipseXThetaArea\t"<<GetEllipseXThetaArea(energy)<<endl;
                    cout<<"SetEllipseYPhiArea\t"<<GetEllipseYPhiArea(energy)<<endl;
                  }
                }
            }
          }
        }
        again=false;
        GateMessage("Physic", 1, "[TPSPencilBeam] Plan description file successfully loaded."<<G4endl);
      }
    inFile.close();

    mTotalNumberOfSpots=mPencilBeams.size();
    if (mTotalNumberOfSpots==0) {
      G4cout<<"ERROR - 0 spots have been loaded from the file \""<<mPlan<<"\" simulation abort!\n"<<G4endl;
      exit (0);
    }

    GateMessage("Physic", 1, "[TPSPencilBeam] Starting particle generation:  "
                << mTotalNumberOfSpots<<" spots loaded."<<G4endl);
    mPDF = new double[mTotalNumberOfSpots];
    for (int i=0; i<mTotalNumberOfSpots; i++){
      // it is strongly adviced to set mFlatGenerationFlag=false
      // a few test demonstrated a lot more efficiency for "real field like" simulation in patients.
      if (mFlatGenerationFlag) {
	mPDF[i]= 1;
      }
      else {
	mPDF[i]=mPencilBeams[i]->GetWeight();
	mPencilBeams[i]->SetWeight(1);
      }
    }
    mDistriGeneral = new RandGeneral(engine,mPDF, mTotalNumberOfSpots, 0);

    //---------INITIALIZATION - END-----------------------
  }

  //---------GENERATION - START-----------------------
  int bin=mTotalNumberOfSpots*mDistriGeneral->fire();
  mPencilBeams[bin]->GenerateVertex(aEvent);
}
//---------GENERATION - END-----------------------

//------------------------------------------------------------------------------------------------------
double GateSourceTPSPencilBeam::ConvertMuToProtons(double weight, double energy)
{
  double K=37.60933;
  double SP=9.6139E-09*pow(energy,4)-7.0508E-06*pow(energy,3)+2.0028E-03*pow(energy,2)-2.7615E-01*pow(energy,1)+2.0082E+01*pow(energy,0);
  double PTP=1;
  double Gain=3./(K*SP*PTP*1.602176E-10);
  return (weight*Gain);
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
void GateSourceTPSPencilBeam::LoadClinicalBeamProperties(){

  const int MAXLINE=256;
  char oneline[MAXLINE];
  int PolOrder;

  ifstream inFile(mSourceDescriptionFile);
  if (! inFile){
    G4cout << "ERROR TPS PENCIL BEAM:  cannot open source description file!\n"<<G4endl;
    exit (0);
  }

  for (int i=0;i<4;i++) inFile.getline(oneline, MAXLINE);
  // distance source patient
  mDistanceSourcePatient=atof(oneline);

  for (int i=0;i<2;i++) inFile.getline(oneline, MAXLINE);
  // distance SMX patient
  mDistanceSMXToIsocenter=atof(oneline);

  for (int i=0;i<2;i++) inFile.getline(oneline, MAXLINE);
  // distance SMY patient
  mDistanceSMYToIsocenter=atof(oneline);

  for (int i=0;i<5;i++) inFile.getline(oneline, MAXLINE);
  // Energy
  PolOrder=atoi(oneline);
  mEnergy.push_back(PolOrder);
  inFile.getline(oneline, MAXLINE);
  for (int i=0;i<=PolOrder;i++) {
    inFile.getline(oneline, MAXLINE);
    mEnergy.push_back(atof(oneline));
  }

  for (int i=0;i<4;i++) inFile.getline(oneline, MAXLINE);
  // Energy
  PolOrder=atoi(oneline);
  mEnergySpread.push_back(PolOrder);
  inFile.getline(oneline, MAXLINE);
  for (int i=0;i<=PolOrder;i++) {
    inFile.getline(oneline, MAXLINE);
    mEnergySpread.push_back(atof(oneline));
  }

  for (int i=0;i<5;i++) inFile.getline(oneline, MAXLINE);
  // X
  PolOrder=atoi(oneline);
  mX.push_back(PolOrder);
  inFile.getline(oneline, MAXLINE);
  for (int i=0;i<=PolOrder;i++) {
    inFile.getline(oneline, MAXLINE);
    mX.push_back(atof(oneline));
  }

  for (int i=0;i<3;i++) inFile.getline(oneline, MAXLINE);
  // Theta
  PolOrder=atoi(oneline);
  mTheta.push_back(PolOrder);
  inFile.getline(oneline, MAXLINE);
  for (int i=0;i<=PolOrder;i++) {
    inFile.getline(oneline, MAXLINE);
    mTheta.push_back(atof(oneline));
  }

  for (int i=0;i<3;i++) inFile.getline(oneline, MAXLINE);
  // Y
  PolOrder=atoi(oneline);
  mY.push_back(PolOrder);
  inFile.getline(oneline, MAXLINE);
  for (int i=0;i<=PolOrder;i++) {
    inFile.getline(oneline, MAXLINE);
    mY.push_back(atof(oneline));
  }

  for (int i=0;i<3;i++) inFile.getline(oneline, MAXLINE);
  // Phi
  PolOrder=atoi(oneline);
  mPhi.push_back(PolOrder);
  inFile.getline(oneline, MAXLINE);
  for (int i=0;i<=PolOrder;i++) {
    inFile.getline(oneline, MAXLINE);
    mPhi.push_back(atof(oneline));
  }

  for (int i=0;i<5;i++) inFile.getline(oneline, MAXLINE);
  // Emittance X Theta
  PolOrder=atoi(oneline);
  mXThetaEmittance.push_back(PolOrder);
  inFile.getline(oneline, MAXLINE);
  for (int i=0;i<=PolOrder;i++) {
    inFile.getline(oneline, MAXLINE);
    mXThetaEmittance.push_back(atof(oneline));
  }

  for (int i=0;i<3;i++) inFile.getline(oneline, MAXLINE);
  // Emittance Y Phi
  PolOrder=atoi(oneline);
  mYPhiEmittance.push_back(PolOrder);
  inFile.getline(oneline, MAXLINE);
  for (int i=0;i<=PolOrder;i++) {
    inFile.getline(oneline, MAXLINE);
    mYPhiEmittance.push_back(atof(oneline));
  }

  if (mTestFlag){
    G4cout<<"DSP "<<mDistanceSourcePatient<<G4endl;
    G4cout<<"SMX "<<mDistanceSMXToIsocenter<<G4endl;
    G4cout<<"SMY "<<mDistanceSMYToIsocenter<<G4endl;
    for (unsigned int i=0; i<mEnergy.size(); i++) G4cout<<"mEnergy\t"<<mEnergy[i]<<G4endl;
    for (unsigned int i=0; i<mEnergySpread.size(); i++) G4cout<<"mEnergySpread\t"<<mEnergySpread[i]<<G4endl;
    for (unsigned int i=0; i<mX.size(); i++) G4cout<<"mX\t"<<mX[i]<<G4endl;
    for (unsigned int i=0; i<mTheta.size(); i++) G4cout<<"mTheta\t"<<mTheta[i]<<G4endl;
    for (unsigned int i=0; i<mY.size(); i++) G4cout<<"mY\t"<<mY[i]<<G4endl;
    for (unsigned int i=0; i<mPhi.size(); i++) G4cout<<"mPhi\t"<<mPhi[i]<<G4endl;
    for (unsigned int i=0; i<mXThetaEmittance.size(); i++) G4cout<<"mXThetaEmittance\t"<<mXThetaEmittance[i]<<G4endl;
    for (unsigned int i=0; i<mYPhiEmittance.size(); i++) G4cout<<"mYPhiEmittance\t"<<mYPhiEmittance[i]<<G4endl;
  }
}
//------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------
double GateSourceTPSPencilBeam::GetEnergy(double energy){
  double val=0;
  for (int i=0; i<=mEnergy[0]; i++){
    val+=mEnergy[i+1]*pow(energy,mEnergy[0]-i);
  }
  return val;

}

//------------------------------------------------------------------------------------------------------
double GateSourceTPSPencilBeam::GetSigmaEnergy(double energy){
  double val=0;
  for (int i=0; i<=mEnergySpread[0]; i++){
    val+=mEnergySpread[i+1]*pow(energy,mEnergySpread[0]-i);
  }
  return val;
}

//------------------------------------------------------------------------------------------------------
double GateSourceTPSPencilBeam::GetSigmaX(double energy) {
  double val=0;
  for (int i=0; i<=mX[0]; i++){
    val+=mX[i+1]*pow(energy,mX[0]-i);
  }
  return val;
}

//------------------------------------------------------------------------------------------------------
double GateSourceTPSPencilBeam::GetSigmaY(double energy) {
  double val=0;
  for (int i=0; i<=mY[0]; i++){
    val+=mY[i+1]*pow(energy,mY[0]-i);
  }
  return val;
}

//------------------------------------------------------------------------------------------------------
double GateSourceTPSPencilBeam::GetSigmaTheta(double energy) {
  double val=0;
  for (int i=0; i<=mTheta[0]; i++){
    val+=mTheta[i+1]*pow(energy,mTheta[0]-i);
  }
  return val;
}

//------------------------------------------------------------------------------------------------------
double GateSourceTPSPencilBeam::GetSigmaPhi(double energy) {
  double val=0;
  for (int i=0; i<=mPhi[0]; i++){
    val+=mPhi[i+1]*pow(energy,mPhi[0]-i);
  }
  return val;
}

//------------------------------------------------------------------------------------------------------
double GateSourceTPSPencilBeam::GetEllipseXThetaArea(double energy) {
  double val=0;
  for (int i=0; i<=mXThetaEmittance[0]; i++){
    val+=mXThetaEmittance[i+1]*pow(energy,mXThetaEmittance[0]-i);
  }
  return val;
}

//------------------------------------------------------------------------------------------------------
double GateSourceTPSPencilBeam::GetEllipseYPhiArea(double energy) {
  double val=0;
  for (int i=0; i<=mYPhiEmittance[0]; i++){
    val+=mYPhiEmittance[i+1]*pow(energy,mYPhiEmittance[0]-i);
  }
  return val;
}

//------------------------------------------------------------------------------------------------------
G4int GateSourceTPSPencilBeam::GeneratePrimaries( G4Event* event )
{
  GateMessage("Beam", 4, "GeneratePrimaries " << event->GetEventID() << G4endl);
  G4int numVertices = 0;
  GenerateVertex( event );
  numVertices++;
  return numVertices;
}
/*
//------------------------------------------------------------------------------------------------------
G4ThreeVector GateSourceTPSPencilBeam::SetGantryRotation(G4ThreeVector v, double theta){
G4double a,b;
a=v[1]*cos(theta)-v[2]*sin(theta);
b=v[1]*sin(theta)+v[2]*cos(theta);
v[1]=a; v[2]=b;
return v;
}
*/
// FUNCTION
//------------------------------------------------------------------------------------------------------
void ReadLineTo3Doubles(double *toto, char *oneline){
  string data = oneline;
  istringstream iss(data);
  string token;
  for (int j=0; j<3; j++){
    getline(iss, token, ' ');
    toto[j]=atof(token.c_str());
    //	G4cout<<"toto "<<toto[j]<<G4endl;
  }
}
#endif
#endif
