/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include <exception>

#include "GateCompressedVoxelOutput.hh"
#include "GateCompressedVoxelOutputMessenger.hh"
#include "GateVoxelCompressor.hh"
#include "GateTrajectoryNavigator.hh"

#include "globals.hh"
#include "G4Run.hh"
#include "G4Step.hh"
#include "G4Event.hh"
#include "GateHit.hh"
#include "GatePhantomHit.hh"
#include "G4VHitsCollection.hh"
#include "G4HCofThisEvent.hh"
#include "G4TrajectoryContainer.hh"
#include "G4Material.hh"
#include "G4VProcess.hh"
#include "G4ios.hh"
#include "G4UImanager.hh"
#include "GatePrimaryGeneratorAction.hh"

#include "G4ParticleDefinition.hh"
#include "G4Positron.hh"
#include "G4GenericIon.hh"

//#include "GateSourceMgr.hh"

#include "G4Navigator.hh"
#include "G4TransportationManager.hh"
#include "GateOutputMgr.hh"
#include "GateCompressedVoxel.hh"
#include "GateCompressedVoxelParameterized.hh"
#include "GateVVolume.hh"


//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateCompressedVoxelOutput::GateCompressedVoxelOutput(const G4String& name,const G4String& phantomName, GateOutputMgr* outputMgr,DigiMode digiMode,GateCompressedVoxelParameterized* inserter) 
  : GateVOutputModule(name,outputMgr,digiMode),
    m_array(new std::valarray<float>),
    m_arraySquare(new std::valarray<float>),
    m_arrayCounts(new std::valarray<unsigned int>),
    m_inserter(inserter),
    m_fileName(" "), // All default output file from all output modules are set to " ".
                     // They are then checked in GateApplicationMgr::StartDAQ, using
                     // the VOutputModule pure virtual method GiveNameOfFile()
    m_uncertainty(false),
    m_phantomName(phantomName)
{
  m_isEnabled = true; // This module is clearly call by the user, so enable is true when constructing output module
  m_outputMessenger = new GateCompressedVoxelOutputMessenger(this);
  m_trajectoryNavigator = new GateTrajectoryNavigator();
  SetVerboseLevel(0);

  // G4cout << "GateCompressedVoxelOutput::GateCompressedVoxelOutput - Constructor entered. Output name " << GetName() << ", phantomName "<< m_phantomName  << Gateendl  << std::flush ;

}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
GateCompressedVoxelOutput::~GateCompressedVoxelOutput() 
{
  delete m_outputMessenger;
  if (nVerboseLevel > 0) G4cout << "GateCompressedVoxelOutput deleting...\n";
  delete m_array;
}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
const G4String& GateCompressedVoxelOutput::GiveNameOfFile()
{
  return m_fileName;
}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
void GateCompressedVoxelOutput::RecordBeginOfAcquisition()
{
  //  G4cout << "GateCompressedVoxelOutput::RecordBeginOfAcquisition - Entered at " << this << " for "<< GetName()  << Gateendl  << std::flush ;

  if (nVerboseLevel > 2)
    G4cout << "GateCompressedVoxelOutput::RecordBeginOfAcquisition - Entered \n";
  
  G4cout<< (*G4Material::GetMaterialTable()) << Gateendl;

}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// write a binary file (4 bytes floats) containing the dose in cGy
void GateCompressedVoxelOutput::RecordEndOfAcquisition()
{
  static const double cGy(gray/100.0);

  // Get initial dimensions and size of the phantom and calculate original voxel volume
  GateVGeometryVoxelReader* theReader        ( m_inserter->GetReader());
  const G4ThreeVector       voxelSize        ( theReader->GetVoxelSize());
  const G4ThreeVector       voxelNumber      ( theReader->GetVoxelNx(), theReader->GetVoxelNy(), theReader->GetVoxelNz());
  const int                 totalVoxelNumber ( int(voxelNumber.x() * voxelNumber.y() * voxelNumber.z()) );
  const double              voxelVolume      ( voxelSize.x() * voxelSize.y() * voxelSize.z() );
  const int                 dimx             ( 1 );
  const int                 dimy             (  int(voxelNumber.x()) );
  const int                 dimz             (  int(voxelNumber.x() * voxelNumber.y()) );

  // Allocate the array to hold dose for every expanded voxel
  std::valarray<float>*     expandedArray(0);
  try{
    expandedArray = new std::valarray<float>;
    expandedArray->resize( totalVoxelNumber );
  }
  catch(...){  G4Exception( "GateCompressedVoxelOutput::RecordEndOfAcquisition", "RecordEndOfAcquisition", FatalException , "No memory for expanded array"); }

  // Process each (compressed) voxel:
  for (unsigned int i=0; i<m_array->size(); i++){
    // a) calculate dose
    const GateCompressedVoxel& gcv     ( theReader->GetCompressor().GetVoxel(i));
    G4Material*                mat     ( (*G4Material::GetMaterialTable())[gcv[6]]);
    double                     density ( mat->GetDensity());
    double                     volume  ( gcv[3] * gcv[4] * gcv[5] * voxelVolume);
    double                     mass    ( density*volume );
    double                     dose    ( (*m_array)[i] / mass );

    // b) put dose value in expanded voxels
    for (int l=gcv[0]; l<gcv[0]+gcv[3]; l++)
      for (int m=gcv[1]; m<gcv[1]+gcv[4]; m++)
	for (int n=gcv[2]; n<gcv[2]+gcv[5]; n++)
	  (*expandedArray)[ l*dimz + m*dimy + n*dimx ] = dose/cGy;

  }

  // Output the dose array
  std::ofstream f;
  f.open(m_fileName, std::ofstream::out | std::ofstream::binary);
  
  for (unsigned int i=0; i< expandedArray->size(); i++)
    f.write( (char*)&(*expandedArray)[i], sizeof(float));
  f.close();
  
  delete expandedArray;

}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
void GateCompressedVoxelOutput::RecordBeginOfRun(const G4Run * )
{
  if (nVerboseLevel > 2)
    G4cout << "GateCompressedVoxelOutput::RecordBeginOfRun\n";
}
//---------------------------------------------------------------------------

//----------------------------------------------------------------------------
void GateCompressedVoxelOutput::RecordEndOfRun(const G4Run * )
{
  if (nVerboseLevel > 2)
    G4cout << "GateCompressedVoxelOutput::RecordEndOfRun\n";
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void GateCompressedVoxelOutput::RecordBeginOfEvent(const G4Event* )
{
  if (nVerboseLevel > 2)
    G4cout << "GateCompressedVoxelOutput::RecordBeginOfEvent\n";
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void GateCompressedVoxelOutput::RecordEndOfEvent(const G4Event* )
{
  GateVGeometryVoxelReader* theReader( m_inserter->GetReader());
  if(!theReader) G4Exception("GateCompressedVoxelOutput::RecordEndOfEvent", "RecordEndOfEvent", FatalException, "No reader.");

  // Set the array size if necessary
  if(m_array->size() == 0) {
    int voxelNumber = theReader->GetCompressor().GetNbOfCopies();
    m_array->resize(voxelNumber);
    if (m_uncertainty){
      m_arraySquare->resize(voxelNumber);
      m_arrayCounts->resize(voxelNumber);
    }
  }
  

  if (nVerboseLevel > 2)
    G4cout << "GateCompressedVoxelOutput::RecordEndOfEvent - Entered for phantom "<< m_phantomName << Gateendl;

  GatePhantomHitsCollection* PHC = GetOutputMgr()->GetPhantomHitCollection();
  G4int NpHits = PHC->entries();

  for (G4int i=0;i<NpHits;i++){

    GatePhantomHit* h    ( (*PHC)[i] );
    G4double        edep ( h->GetEdep() );
    G4int           n    ( h->GetVoxelCoordinates() );
    G4String        physVolName  ( h->GetPhysVolName() );

    if( 0 == physVolName.compare(0, m_phantomName.size(), m_phantomName) ){

      //  G4cout << "GateCompressedVoxelOutput::RecordEndOfEvent - HIT at voxel "<< n  << " in "<< physVolName << Gateendl;
      
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
	       << Gateendl;
      
    }// end if phantom
  }//end for loop
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void GateCompressedVoxelOutput::RecordStepWithVolume(const GateVVolume *, const G4Step* )
{
  if (nVerboseLevel > 2)
    G4cout << "GateCompressedVoxelOutput::RecordStep\n";
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void GateCompressedVoxelOutput::SetVerboseLevel(G4int val) { 
  nVerboseLevel = val; 
  if (m_trajectoryNavigator) m_trajectoryNavigator->SetVerboseLevel(val);
}
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
void GateCompressedVoxelOutput::SetSaveUncertainty(G4bool b) { 
  m_uncertainty=b; 
}
//---------------------------------------------------------------------------
