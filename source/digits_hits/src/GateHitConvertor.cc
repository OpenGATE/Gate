/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateHitConvertor.hh"

#include "G4UnitsTable.hh"

#include "GateHitConvertorMessenger.hh"
#include "GateTools.hh"
#include "GateDigitizer.hh"
#include "GateConfiguration.h"

const G4String GateHitConvertor::theOutputAlias = "Hits";


GateHitConvertor* GateHitConvertor::GetInstance()
{
  static GateHitConvertor* theHitConvertor = 0;

  if (!theHitConvertor)
    theHitConvertor = new GateHitConvertor();

  return theHitConvertor;
}

GateHitConvertor::GateHitConvertor()
  : GateClockDependent("digitizer/convertor",false)
{
  m_messenger = new GateHitConvertorMessenger(this);
}




GateHitConvertor::~GateHitConvertor()
{
  delete m_messenger;
}



GatePulseList* GateHitConvertor::ProcessHits(const GateCrystalHitsCollection* hitCollection)
{
  if (!hitCollection)
    return 0;

  size_t n_hit = hitCollection->entries();
  if (nVerboseLevel==1)
      G4cout << "[GateHitConvertor::ProcessHits]: "
      	      	"processing hit-collection with " << n_hit << " entries\n";
  if (!n_hit)
    return 0;


  GatePulseList* pulseList = new GatePulseList(GetObjectName());

  size_t i;
  for (i=0;i<n_hit;i++) {
        if (nVerboseLevel>1)
      		G4cout << "[GateHitConvertor::ProcessHits]: processing hit[" << i << "]\n";
      	ProcessOneHit( (*hitCollection)[i], pulseList);
  }

  if (nVerboseLevel==1) {
      G4cout << "[GateHitConvertor::ProcessHits]: returning pulse-list with " << pulseList->size() << " entries\n";
      for (i=0; i<pulseList->size(); i++)
      	G4cout << *((*pulseList)[i]) << G4endl;
      G4cout << G4endl;
  }

  GateDigitizer::GetInstance()->StorePulseList(pulseList);
  GateDigitizer::GetInstance()->StorePulseListAlias(GetOutputAlias(),pulseList);

  return pulseList;
}

void GateHitConvertor::ProcessOneHit(const GateCrystalHit* hit,GatePulseList* pulseList)
{
 if (hit->GetEdep()==0) {
    if (nVerboseLevel>1)
      	G4cout << "[GateHitConvertor::ProcessOneHit]: energy is null for " << *hit << " -> hit ignored\n\n";
    return;
  }

  GatePulse* pulse = new GatePulse(hit);

  pulse->SetRunID( hit->GetRunID() );

  	//  G4cout << "HitConvertor : runID = " << hit->GetRunID() << G4endl;


  pulse->SetEventID( hit->GetEventID() );
  pulse->SetSourceID( hit->GetSourceID() );
  pulse->SetSourcePosition( hit->GetSourcePosition() );
  pulse->SetTime( hit->GetTime() );
  pulse->SetEnergy( hit->GetEdep() );
  pulse->SetLocalPos( hit->GetLocalPos() );
  pulse->SetGlobalPos( hit->GetGlobalPos() );
  pulse->SetPDGEncoding( hit->GetPDGEncoding() );
  pulse->SetOutputVolumeID( hit->GetOutputVolumeID() );
  pulse->SetNPhantomCompton( hit->GetNPhantomCompton() );
  pulse->SetNCrystalCompton( hit->GetNCrystalCompton() );
  pulse->SetNPhantomRayleigh( hit->GetNPhantomRayleigh() );
  pulse->SetNCrystalRayleigh( hit->GetNCrystalRayleigh() );
  pulse->SetComptonVolumeName( hit->GetComptonVolumeName() );
  pulse->SetRayleighVolumeName( hit->GetRayleighVolumeName() );
  pulse->SetVolumeID( hit->GetVolumeID() );
  pulse->SetScannerPos( hit->GetScannerPos() );
  pulse->SetScannerRotAngle( hit->GetScannerRotAngle() );
#ifdef GATE_USE_OPTICAL
  pulse->SetOptical( hit->GetPDGEncoding() == 0 );
#endif
  pulse->SetNSeptal( hit->GetNSeptal() );  // HDS : septal penetration

  if (hit->GetComptonVolumeName().empty()) {
    pulse->SetComptonVolumeName( "NULL" );
    pulse->SetSourceID( -1 );
  }

  if (hit->GetRayleighVolumeName().empty()) {
    pulse->SetRayleighVolumeName( "NULL" );
    pulse->SetSourceID( -1 );
  }

  if (nVerboseLevel>1)
      	G4cout << "[GateHitConvertor::ProcessOneHit]: " << G4endl
	       << "\tprocessed " << *hit << G4endl
	       << "\tcreated new pulse:" << G4endl
	       << *pulse << G4endl;

  pulseList->push_back(pulse);
}

void GateHitConvertor::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Convert hits into pulses for '" << GateDigitizer::GetInstance()->GetObjectName() << "'" << G4endl;
}
