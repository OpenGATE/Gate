/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/
// GND 2022 Class to remove
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



GatePulseList* GateHitConvertor::ProcessHits(const GateHitsCollection* hitCollection)
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

        if((*hitCollection)[i]->GetEdep()==0){
            if (nVerboseLevel>1)
                G4cout << "[GateHitConvertor::ProcessOneHit]: energy is null for " << *(*hitCollection)[i] << " -> hit ignored\n\n";
        }
        else{
            ProcessOneHit( (*hitCollection)[i], pulseList);
        }


  }

 if (nVerboseLevel>1) {
      G4cout << "[GateHitConvertor::ProcessHits]: returning pulse-list with " << pulseList->size() << " entries\n";
      for (i=0; i<pulseList->size(); i++)
        G4cout << *((*pulseList)[i]) << Gateendl;
      G4cout << Gateendl;
  }

  GateDigitizer::GetInstance()->StorePulseList(pulseList);
  GateDigitizer::GetInstance()->StorePulseListAlias(GetOutputAlias(),pulseList);

  return pulseList;
}


 GatePulseList* GateHitConvertor::ProcessHits(std::vector<GateHit*> vhitCollection){

    size_t n_hit = vhitCollection.size();
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
          //Here no problem
         // ProcessOneHit( (std::make_shared<GateHit>(vhitCollection.at(i))).get(), pulseList);
          ProcessOneHit(vhitCollection.at(i), pulseList);
    }

   if (nVerboseLevel==1) {
        G4cout << "[GateHitConvertor::ProcessHits]: returning pulse-list with " << pulseList->size() << " entries\n";
        for (i=0; i<pulseList->size(); i++)
          G4cout << *((*pulseList)[i]) << Gateendl;
        G4cout << Gateendl;
    }

    GateDigitizer::GetInstance()->StorePulseList(pulseList);
    GateDigitizer::GetInstance()->StorePulseListAlias(GetOutputAlias(),pulseList);

    return pulseList;
}

void GateHitConvertor::ProcessOneHit(const GateHit* hit,GatePulseList* pulseList)
{
 /*if (hit->GetEdep()==0) {
    if (nVerboseLevel>1)
      	G4cout << "[GateHitConvertor::ProcessOneHit]: energy is null for " << *hit << " -> hit ignored\n\n";
    return;
  }*/

  GatePulse* pulse = new GatePulse(hit);

  pulse->SetRunID( hit->GetRunID() );

      //G4cout << "HitConvertor : eventID = " << hit->GetEventID() << Gateendl;
      //G4cout << "HitConvertor : edep = " << hit->GetEdep() << Gateendl;


  pulse->SetEventID( hit->GetEventID() );
  pulse->SetSourceID( hit->GetSourceID() );
  pulse->SetSourcePosition( hit->GetSourcePosition() );
  pulse->SetTime( hit->GetTime() );
  pulse->SetEnergy( hit->GetEdep() );
  pulse->SetMaxEnergy( hit->GetEdep() );
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
  pulse->SetOptical( hit->GetPDGEncoding() == -22 );
#endif
  pulse->SetNSeptal( hit->GetNSeptal() );  // HDS : septal penetration

  // AE : Added for IdealComptonPhot adder which take into account several Comptons in the same volume
  pulse->SetPostStepProcess(hit->GetPostStepProcess());
  pulse->SetEnergyIniTrack(hit->GetEnergyIniTrack());
  pulse->SetEnergyFin(hit->GetEnergyFin());
  pulse->SetProcessCreator(hit->GetProcess());
  pulse->SetTrackID(hit->GetTrackID());
  pulse->SetParentID(hit->GetParentID());
  pulse->SetSourceEnergy(hit->GetSourceEnergy());
  pulse->SetSourcePDG(hit->GetSourcePDG());
  pulse->SetNCrystalConv( hit->GetNCrystalConv() );


    
//-------------------------------------------------
     


  if (hit->GetComptonVolumeName().empty()) {
    pulse->SetComptonVolumeName( "NULL" );
    pulse->SetSourceID( -1 );
  }

  if (hit->GetRayleighVolumeName().empty()) {
    pulse->SetRayleighVolumeName( "NULL" );
    pulse->SetSourceID( -1 );
  }

  if (nVerboseLevel>1)
      	G4cout << "[GateHitConvertor::ProcessOneHit]: \n"
	       << "\tprocessed " << *hit << Gateendl
	       << "\tcreated new pulse:\n"
	       << *pulse << Gateendl;

  pulseList->push_back(pulse);
}

void GateHitConvertor::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Convert hits into pulses for '" << GateDigitizer::GetInstance()->GetObjectName() << "'\n";
}
