/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \brief Class GateWashOutActor
*/

//-----------------------------------------------------------------------------
/*  GateWashOutActor :: The inclusion of the WashOut effect is implemeneted 
    in this class. The model described in Mizuno et al PMB 48 (2003) has been 
    incorporated. This model is expressed in terms of three exponential 
    components (slow, medium and fast) and its corresponding parameters. */
//-----------------------------------------------------------------------------

#ifndef GATEWASHOUTACTOR_CC
#define GATEWASHOUTACTOR_CC

#include "GateWashOutActor.hh"
#include "GateMiscFunctions.hh"
#include "GateClock.hh"


//-----------------------------------------------------------------------------
GateWashOutActor::GateWashOutActor(G4String name, G4int depth):
  GateVActor(name,depth) {
    
  GateDebugMessage("Actor",4,"GateWashOutActor() -- begin" << G4endl);
  
  pActor = new GateActorMessenger(this);
  pWashOutActor = new GateWashOutActorMessenger(this);
  
  GateDebugMessage("Actor",4,"GateWashOutActor() -- end" << G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateWashOutActor::~GateWashOutActor() 
{
  GateMessage("Actor",4,"~GateWashOutActor() -- begin" << G4endl);
  
  delete pActor;
  delete pWashOutActor;

  GateMessage("Actor",4,"~GateWashOutActor() -- end" << G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateWashOutActor::Construct()
{
  GateVActor::Construct();

  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnableEndOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnableUserSteppingAction(true);
  EnablePostUserTrackingAction(false);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateWashOutActor::BeginOfRunAction(const G4Run * r) {
  GateVActor::BeginOfRunAction(r);

  G4double ActWashOut;   
  mGateWashOutActivityIni.clear(); 

  for(G4int nsource= 0 ; nsource<GateSourceMgr::GetInstance()->GetNumberOfSources() ; nsource++ ) {
    
    GateVSource * SourceIni = (GateSourceMgr::GetInstance())->GetSource(nsource); 
    
    if ( SourceIni->GetType() == G4String("gps") ) {
      ActWashOut = SourceIni->GetActivity(); }

    else if ( SourceIni->GetType() == G4String("") ) {
      GateSourceVoxellized * SourceVoxlIni = (GateSourceVoxellized *) SourceIni;
      GateVSourceVoxelReader * VSReaderIni = SourceVoxlIni->GetReader();      
      ActWashOut = VSReaderIni->GetTempTotalActivity(); }
      
    else{
      GateError("WashOut Actor :: ERROR: Source Type non recognised." << G4endl); }        
      
    mGateWashOutActivityIni.push_back(ActWashOut);  
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateWashOutActor::BeginOfEventAction(const G4Event * event)
{
  GateMessage("Actor", 4, "GateWashOutActor -- Begin of Event" << G4endl);
  GateVActor::BeginOfEventAction(event);
  
  mSourceID = (GateSourceMgr::GetInstance())->GetCurrentSourceID();  // Source ID       
  mSourceNow = (GateSourceMgr::GetInstance())->GetSource(mSourceID);  
  
  if ( mSourceNow->GetType() == G4String("") ) {
    GateSourceVoxellized * SourceVoxlNow = (GateSourceVoxellized *) mSourceNow; // Voxelized source
    mSVReader = SourceVoxlNow->GetReader(); } 
   
  mTimeNow = (GateSourceMgr::GetInstance())->GetTime(); // Present time
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateWashOutActor::EndOfEventAction(const G4Event * event)
{
  GateMessage("Actor", 4, "GateWashOutActor -- End of Event" << G4endl);
  GateVActor::EndOfEventAction(event);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateWashOutActor::PreUserTrackingAction(const GateVVolume * /*v*/, const G4Track* /*t*/)
{
  mWashOutIsFistStep = true; // First step of the particle 
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateWashOutActor::UserSteppingAction(const GateVVolume * vol, const G4Step* step) {
 GateVActor::UserSteppingAction(vol,step);

  if ( !mWashOutIsFistStep ) return; 
 
  if ( mWashOutIsFistStep && step->GetTrack()->GetTrackID() == 1 ) {  // Primary particle, first step
   
    G4String ParticleNameNow = step->GetTrack()->GetParticleDefinition()->GetParticleName(); // Particle name

 //   if ( ParticleNameNow == G4String("e+") ) { 

      G4String MatNameNow = step->GetPreStepPoint()->GetMaterial()->GetName();  // Present material name
   
      for ( G4int iRow = 0; iRow<(G4int)mGateWashOutMaterials.size(); iRow++ ) { 

        G4String MatName = mGateWashOutMaterials[iRow]; // WashOut materials
         
          if ( MatName == MatNameNow ) { 

            G4double Model = mGateWashOutParemeters[iRow][0] * exp( - (mTimeNow * log(2.0) / mGateWashOutParemeters[iRow][1] ) ) +  // First component
            mGateWashOutParemeters[iRow][2] * exp( - (mTimeNow * log(2.0) / mGateWashOutParemeters[iRow][3] ) ) +  // Second component
            mGateWashOutParemeters[iRow][4] * exp( - (mTimeNow * log(2.0) / mGateWashOutParemeters[iRow][5] ) );  // Third component  
    
            G4double ActivityNew = mGateWashOutActivityIni[mSourceID] * Model;  // Apply the washout model

            if ( mSourceNow->GetType() == G4String("") ) {    // Set the total activity of the source (with the washout decay)
              mSVReader->SetTempTotalActivity( ActivityNew ); 
	    }  
            else if ( mSourceNow->GetType() == G4String("gps") ) {      
              mSourceNow->SetActivity( ActivityNew ); 
	    }

//            G4cout << "##############################################################################################"  << G4endl;
//            G4cout << "######### WashOutActor Step :: Source ID = " << mSourceID << G4endl;
//            G4cout << "######### WashOutActor Step :: Source Type = " << mSourceNow->GetType() << G4endl;
//            G4cout << "######### WashOutActor Step :: Particle at Position = " << ParticleNameNow << " at " << step->GetPreStepPoint()->GetPosition() << G4endl;	      
//            G4cout << "######### WashOutActor Step :: Current Material for this Event = " << MatNameNow << G4endl;
//            G4cout << "######### WashOutActor Step :: Time for this Event (s) = " << mTimeNow/s << G4endl;
//            G4cout << "######### WashOutActor Step :: Initial Activity (Bq) = " << mGateWashOutActivityIni[mSourceID]/becquerel << G4endl;      	    
//            G4cout << "######### WashOutActor Step :: Washout Model value = " << Model << G4endl;         
//            G4cout << "######### WashOutActor Step :: Activity after modification (Bq) = " << ActivityNew/becquerel << G4endl;  
//            G4cout << "##############################################################################################"  << G4endl;
    
   //    }	
      }  
    }
    mWashOutIsFistStep = false; 
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/*  GateWashOutActor::ReadWashOutTable(G4String fileName) reads the washout 
    parameters from a file (fileName).The model described in Mizuno et al 
    PMB 48 (2003) has been incorporated. This model is expressed in terms
    of three components (slow, medium and fast; order not important). 
    For each component and for each material, its fraction and its 
    half life (with units) are required. The sum of the ratios has to be equal 
    to the unity. Example (in fileName):
    Brain   0.35  10000 s    0.3  140 s    0.35  2.0 s  */
//-----------------------------------------------------------------------------
void GateWashOutActor::ReadWashOutTable(G4String fileName)
{        
  mGateWashOutMaterials.clear();
  mGateWashOutParemeters.clear();
  
  std::ifstream inFile;

  GateMessage("Actor", 0, "GateWashOutActor -- ReadWashOutTable : fileName: " << fileName << G4endl;);
  
  inFile.open(fileName.c_str(),std::ios::in);

  if (inFile.is_open()){
    G4int nTotCol;
    inFile >> nTotCol;
  
    G4String matWashOut;
    std::vector<G4double> parWashOut (6); 
    
    G4String unit;
    G4double diff=1e-8;
    
    for ( G4int iRow=0; iRow<nTotCol; iRow++ ) {

      inFile >> matWashOut; 
      
      inFile >> parWashOut[0] >> parWashOut[1] >> unit; 
      parWashOut[1] = ScaleValue(parWashOut[1],unit);   
      
      inFile >> parWashOut[2] >> parWashOut[3] >> unit;
      parWashOut[3] = ScaleValue(parWashOut[3],unit);  
      
      inFile >> parWashOut[4] >> parWashOut[5] >> unit; 
      parWashOut[5] = ScaleValue(parWashOut[5],unit);   
      
      G4double sumWashOut = parWashOut[0] + parWashOut[2] + parWashOut[4];
      
      if ( fabs(sumWashOut-1.0) > diff ) {
        GateError("WashOut Actor :: ERROR: Total WashOut Ratio non equal to 1." << G4endl);
      }  

// Print information
      GateMessage("Actor", 0, 
              "Line " << iRow+1 << " of " << nTotCol << G4endl <<
              "\tMaterial: " << matWashOut << G4endl <<
              "\tRatio First WashOut Component: " << parWashOut[0] << G4endl <<                 
              "\tHalf Life First WashOut Component (ns): " << parWashOut[1] << G4endl <<              
              "\tRatio Second WashOut Component: " << parWashOut[2] << G4endl <<    
              "\tHalf Life Second WashOut Component (ns): " << parWashOut[3] << G4endl <<                   
              "\tRatio Third WashOut Component: " << parWashOut[4] << G4endl <<   
              "\tHalf Life Third WashOut Component (ns): " << parWashOut[5] << G4endl);    

      mGateWashOutMaterials.push_back(matWashOut);
      mGateWashOutParemeters.push_back(parWashOut);
     
    }
  }
  
  else {

    GateError("WashOut Actor :: ERROR in opening/reading WashOut datafile." << G4endl);

    inFile.close();
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4double GateWashOutActor::ScaleValue(G4double value,G4String unit)
{
  double res = 0.;
  
  if(unit=="s")  res = value * 1e+09;
  if(unit=="ms") res = value * 1e+06;
  if(unit=="mus") res = value * 1e+03;
  if(unit=="ns") res = value * 1;
  if(unit=="ps") res = value * 1e-03;
  
  return res;
}
//-----------------------------------------------------------------------------


#endif /* end #define GATEWASHOUTACTOR_CC */
