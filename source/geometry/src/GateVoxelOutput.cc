/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "globals.hh"
#include "G4Run.hh"
#include "G4Step.hh"
#include "G4Event.hh"
#include "G4VHitsCollection.hh"
#include "G4HCofThisEvent.hh"
#include "G4TrajectoryContainer.hh"
#include "G4Material.hh"
#include "G4VProcess.hh"
#include "G4ios.hh"
#include "G4UImanager.hh"
#include "G4RunManager.hh"
#include "G4ParticleDefinition.hh"
#include "G4Positron.hh"
#include "G4GenericIon.hh"
#include "G4Navigator.hh"
#include "G4TransportationManager.hh"
#include "G4Material.hh"

//#include "GateSourceMgr.hh"
#include "GatePrimaryGeneratorAction.hh"
#include "GateVoxelOutput.hh"
#include "GateVoxelOutputMessenger.hh"
#include "GateTrajectoryNavigator.hh"
#include "GateOutputMgr.hh"
#include "GateVoxelBoxParameterized.hh"
#include "GateRegularParameterized.hh"
#include "GateFictitiousVoxelMapParameterized.hh"
#include "GateCrystalHit.hh"
#include "GatePhantomHit.hh"
#include "GateRecorderBase.hh"

//--------------------------------------------------------------------------------------------------
GateVoxelOutput::GateVoxelOutput(const G4String& name,const G4String& phantomName, GateOutputMgr* outputMgr,DigiMode digiMode,GateVoxelBoxParameterized* inserter) 
  : GateVOutputModule(name,outputMgr,digiMode),
    m_array(new std::valarray<float>),
    m_arraySquare(new std::valarray<float>),
    m_arrayCounts(new std::valarray<unsigned int>),
    m_voxelInserter(inserter),
    m_fileName(" "), // All default output file from all output modules are set to " ".
                     // They are then checked in GateApplicationMgr::StartDAQ, using
                     // the VOutputModule pure virtual method GiveNameOfFile()
    m_uncertainty(false),
    m_phantomName(phantomName)
{
  m_isEnabled = true; // This module is clearly call by the user, so enable is true when constructing output module
  m_outputMessenger = new GateVoxelOutputMessenger(this);
  m_trajectoryNavigator = new GateTrajectoryNavigator();
  SetVerboseLevel(0);

  m_inserterType=1;  // The constructor is called with a voxelBoxParameterized

}
//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
GateVoxelOutput::GateVoxelOutput(const G4String& name,const G4String& phantomName, GateOutputMgr* outputMgr,DigiMode digiMode,GateRegularParameterized* inserter) 
  : GateVOutputModule(name,outputMgr,digiMode),
    m_array(new std::valarray<float>),
    m_arraySquare(new std::valarray<float>),
    m_arrayCounts(new std::valarray<unsigned int>),
    m_regularInserter(inserter),
    m_fileName(" "), // All default output file from all output modules are set to " ".
                     // They are then checked in GateApplicationMgr::StartDAQ, using
                     // the VOutputModule pure virtual method GiveNameOfFile()
    m_uncertainty(false),
    m_phantomName(phantomName)
{
  m_isEnabled = true; // This module is clearly call by the user, so enable is true when constructing output module
  m_outputMessenger = new GateVoxelOutputMessenger(this);
  m_trajectoryNavigator = new GateTrajectoryNavigator();
  SetVerboseLevel(0);

  m_inserterType=2;  // The constructor is called with a regularParameterized
}
//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
GateVoxelOutput::GateVoxelOutput(const G4String& name,const G4String& phantomName, GateOutputMgr* outputMgr,DigiMode digiMode,GateFictitiousVoxelMapParameterized* inserter) 
  : GateVOutputModule(name,outputMgr,digiMode),
    m_array(new std::valarray<float>),
    m_arraySquare(new std::valarray<float>),
    m_arrayCounts(new std::valarray<unsigned int>),
    m_fictitiousInserter(inserter),
    m_fileName(" "), // All default output file from all output modules are set to " ".
                     // They are then checked in GateApplicationMgr::StartDAQ, using
                     // the VOutputModule pure virtual method GiveNameOfFile()
    m_uncertainty(false),
    m_phantomName(phantomName)
{
  m_isEnabled = true; // This module is clearly call by the user, so enable is true when constructing output module
  m_outputMessenger = new GateVoxelOutputMessenger(this);
  m_trajectoryNavigator = new GateTrajectoryNavigator();
  SetVerboseLevel(0);

  m_inserterType=3;  // The constructor is called with a FictitiousVoxelMapParameterized
}
//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
GateVoxelOutput::~GateVoxelOutput() 
{
  delete m_outputMessenger;
  if (nVerboseLevel > 0) G4cout << "GateVoxelOutput deleting..." << G4endl;
  delete m_array;
}
//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
const G4String& GateVoxelOutput::GiveNameOfFile()
{
  return m_fileName;
}
//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
void GateVoxelOutput::RecordBeginOfAcquisition()
{
  //  G4cout << "GateVoxelOutput::RecordBeginOfAcquisition - Entered at " << this << " for "<< GetName()  << G4endl  << std::flush ;

  if (nVerboseLevel > 2)
    G4cout << "GateVoxelOutput::RecordBeginOfAcquisition - Entered " << G4endl;
  
  G4cout<< (*G4Material::GetMaterialTable()) << G4endl;
}
//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
// write a binary file (4 bytes floats) containing the dose in cGy
void GateVoxelOutput::RecordEndOfAcquisition()
{
  static const double Gy (joule/kilogram);
  static const double cGy(Gy/100.0);

  // G4cout << "GateVoxelOutput::RecordEndOfAcquisition - Entered at " << this << " for "<< GetName()  << G4endl  << std::flush ;
  
  if (nVerboseLevel > 0)
    G4cout << "GateVoxelOutput::RecordEndOfAcquisition - Writing to "<< m_fileName << G4endl;
  
  std::ofstream f;
  
  GateVGeometryVoxelReader* theReader=0;
  switch ( m_inserterType )
  {
    case 1:
      theReader = m_voxelInserter->GetReader();
      break;
    case 2:
      theReader = m_regularInserter->GetReader();
      break;
    case 3:
      theReader = m_fictitiousInserter->GetReader();
      break;
    default:
      G4Exception ( "GateVoxelOutput::RecordEndOfAcquisition()", "Reader type not defined", FatalException,
                    "Aborting." );
  }

//  GateVGeometryVoxelReader* theReader( m_inserter->GetReader());
  if(!theReader) G4Exception( "GateVoxelOutput::RecordEndOfAcquisition", "RecordEndOfAcquisition", FatalException, "No reader" );
  
  const double voxelSize ( theReader->GetVoxelSize().x() * 
			   theReader->GetVoxelSize().y() * 
			   theReader->GetVoxelSize().z()
			   );
  
  // Output the dose collection (main file)
  f.open(m_fileName, std::ofstream::out | std::ofstream::binary);
  
  for (unsigned int i=0; i<m_array->size(); i++){
    double mass ( voxelSize * theReader->GetVoxelMaterial(i)->GetDensity() );
    float  dose ( (*m_array)[i] / mass / cGy );   // float to get a 32 bits floating point number
    f.write( (char*)&dose, sizeof(dose));
    
    if (nVerboseLevel >0 )
      G4cout << "bin " << i
	     << ", energy "<< (*m_array)[i] 
	     << ", dose "<<   dose 
	     << G4endl;
  }
  f.close();
  
  // Output the uncertainty file if requested
  // The relative error on dose is calculated with the folowing formula:
  // 
  //              /                                     \ ^1/2
  //              |      N*Sum( di^2 )  -  Sum^2(di)    |
  //  relError =  |   ________________________________  |
  //              |                                     |
  //              \           (N-1)*Sum^2(di)           /
  //
  //   where di represents the energy deposit in one hit and N the number of energy deposits (hits)
  
  if (m_uncertainty){
    f.open( (m_fileName+"U").c_str(), std::ofstream::out | std::ofstream::binary);
    
    for (unsigned int i=0; i<m_array->size(); i++){
      float relativeError(0) ;
      float relativeErrorSquared(0);
      
      // Safeguard check against nan's
      if ((*m_arrayCounts)[i]>1 && (*m_array)[i] !=0 ){
	double N ( (*m_arrayCounts)[i]                );
	double SS( (*m_array)[i] * (*m_array)[i]      );
	double S2( (*m_arraySquare)[i]                );
	relativeErrorSquared = ( N*S2 - SS )/ ( (N-1)*SS );
	if ( abs(relativeErrorSquared) < 1.0e-15 ) relativeErrorSquared=0; // Chop tiny values 
	relativeError=sqrt(relativeErrorSquared);
      }
      
      f.write( (char*)&relativeError, sizeof(relativeError));
      
      if (nVerboseLevel >0 )
	G4cout << "bin " << i
	       << ", sum of squares " << (*m_arraySquare)[i]
	       << ", square of sum  " << ( (*m_array)[i]*(*m_array)[i] )
	       << ", counts "         << (*m_arrayCounts)[i]
	       << ", relative error " << relativeError
	       << G4endl;
    }
    f.close();
  }
  
}
//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
void GateVoxelOutput::RecordBeginOfRun(const G4Run * )
{
  if (nVerboseLevel > 2)
    G4cout << "GateVoxelOutput::RecordBeginOfRun" << G4endl;
}
//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
void GateVoxelOutput::RecordEndOfRun(const G4Run * )
{
  if (nVerboseLevel > 2)
    G4cout << "GateVoxelOutput::RecordEndOfRun" << G4endl;
}
//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
void GateVoxelOutput::RecordBeginOfEvent(const G4Event* )
{
  if (nVerboseLevel > 2)
    G4cout << "GateVoxelOutput::RecordBeginOfEvent" << G4endl;
}
//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
void GateVoxelOutput::RecordEndOfEvent(const G4Event* )
{
  GateVGeometryVoxelReader* theReader=0;
  switch ( m_inserterType )
  {
    case 1:
      theReader = m_voxelInserter->GetReader();
      break;
    case 2:
      theReader = m_regularInserter->GetReader();
      break;
    case 3:
      theReader = m_fictitiousInserter->GetReader();
      break;
    default:
      G4Exception ( "GateVoxelOutput::RecordEndOfAcquisition()", "Reader type not defined", FatalException,
                    "Aborting." );
  }

//  GateVGeometryVoxelReader* theReader( m_inserter->GetReader());
  if(!theReader) G4Exception( "GateVoxelOutput::RecordEndOfAcquisition", "RecordEndOfAcquisition", FatalException, "No reader.");

  // Set the array size if necessary
  if(m_array->size() == 0) {
    int voxelNumber = theReader->GetVoxelNx() * theReader->GetVoxelNy() * theReader->GetVoxelNz();
    m_array->resize(voxelNumber);
    if (m_uncertainty){
      m_arraySquare->resize(voxelNumber);
      m_arrayCounts->resize(voxelNumber);
    }
  }
  

  if (nVerboseLevel > 2)
    G4cout << "GateVoxelOutput::RecordEndOfEvent - Entered for phantom "<< m_phantomName << G4endl;

  GatePhantomHitsCollection* PHC = GetOutputMgr()->GetPhantomHitCollection();
  G4int NpHits = PHC->entries();

  for (G4int i=0;i<NpHits;i++){

    GatePhantomHit* h    ( (*PHC)[i] );
    G4double        edep ( h->GetEdep() );
    G4int           n    ( h->GetVoxelCoordinates() );
    G4String        physVolName  ( h->GetPhysVolName() );

    if( 0 == physVolName.compare(0, m_phantomName.size(), m_phantomName) ){

      //  G4cout << "GateVoxelOutput::RecordEndOfEvent - HIT at voxel "<< n  << " in "<< physVolName << G4endl;
      
      (*m_array)[n]+=edep;
      if (m_uncertainty){
	(*m_arrayCounts) [n]++;
	(*m_arraySquare) [n]+= edep*edep;
      }

      if (nVerboseLevel > 2) 
	G4cout << "hit= " << i 
	       << ", n= " << n 
	       << ", edep " << edep
	       << ", process "   <<  h->GetProcess()
	       << ", array[n] "  <<  (*m_array)[n]
	       << ", square[n] " <<  (*m_arraySquare)[n]
	       << ", counts[n] " <<  (*m_arrayCounts)[n]
	       << G4endl;
      
    }// end if phantom    
  }//end for loop
}
//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
void GateVoxelOutput::RecordStepWithVolume(const GateVVolume *, const G4Step* )
{
  if (nVerboseLevel > 2)
    G4cout << "GateVoxelOutput::RecordStep" << G4endl;
}
//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
void GateVoxelOutput::SetVerboseLevel(G4int val) { 
  nVerboseLevel = val; 
  if (m_trajectoryNavigator) m_trajectoryNavigator->SetVerboseLevel(val);
}
//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
void GateVoxelOutput::SetSaveUncertainty(G4bool b) { 
  m_uncertainty=b; 
 }
//--------------------------------------------------------------------------------------------------


