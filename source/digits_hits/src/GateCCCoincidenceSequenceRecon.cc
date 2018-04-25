/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateCCCoincidenceSequenceRecon.hh"
#include "G4UnitsTable.hh"
#include "GateCCCoincidenceSequenceReconMessenger.hh"
#include "GateTools.hh"
#include "GateVolumeID.hh"
#include "GateOutputVolumeID.hh"
#include "GateVVolume.hh"
#include "GateDetectorConstruction.hh"
#include "GateCrystalSD.hh"
#include "GateObjectChildList.hh"
#include "GateMaps.hh"





GateCCCoincidenceSequenceRecon::GateCCCoincidenceSequenceRecon(GateCoincidencePulseProcessorChain* itsChain,
			   const G4String& itsName)
  : GateVCoincidencePulseProcessor(itsChain,itsName),
    m_sequencePolicy(kSinglesTime)
{

  m_messenger = new GateCCCoincidenceSequenceReconMessenger(this);


}

GateCCCoincidenceSequenceRecon::~GateCCCoincidenceSequenceRecon()
{
  delete m_messenger;
}




GateCoincidencePulse* GateCCCoincidenceSequenceRecon::ProcessPulse(GateCoincidencePulse* inputPulse, G4int )
{
  if (!inputPulse ){
      if (nVerboseLevel>1)
      	G4cout << "[GateCCCoincidenceSequenceRecon::ProcessOnePulse]: input pulse was null -> nothing to do\n\n";
      return 0;
  }
  //GateCoincidencePulse* outputPulse=0;
   GateCoincidencePulse* outputPulse= new GateCoincidencePulse(*inputPulse);
  //inputPulse->size()!=2) {

 // unsigned long long int  currentTime = (unsigned long long int)(inputPulse->GetTime()/picosecond);
//If I want to throw the pulse I should  returnoutputPulse=0;



  switch(m_sequencePolicy) {
    case kSinglesTime :
      // G4cout << '2';
      //The coincidence sorter has orderer the singles by  time to create the coincidence
      // outputPulse= new GateCoincidencePulse(*inputPulse);
      break;
    case kLowestEnergyFirst :
      sort( outputPulse->begin( ), outputPulse->end( ), [ ]( const GatePulse& pulse1, const GatePulse& pulse2 )
      {
         return pulse1.GetEnergy() < pulse2.GetEnergy();
      });

      break;
    case kRandomly :
      //difference entre random_suffle and shuffle
      seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::shuffle ( outputPulse->begin( ), outputPulse->end( ), std::default_random_engine(seed));
     // G4cout<<"seed "<<seed <<"engine "<<std::default_random_engine(seed)<<G4endl;
      break;
    case kSpatially :

      break;
    case kRevanC_CSR:
      G4cout << '2';
      break;
   default:
        std::cout << "default NOT sequence policy taken\n";
       break;
  }



  return outputPulse;
}





void GateCCCoincidenceSequenceRecon::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Sequence policy " <<m_sequencePolicy  << Gateendl;
}




void  GateCCCoincidenceSequenceRecon::SetSequencePolicy(const G4String& policy)
{
//    if (policy=="singlesTime ")
//        m_sequencePolicy=kSinglesTime;
    //SinglesTime set  in the constructor
     if (policy=="lowestEnergyFirst")
        m_sequencePolicy=kLowestEnergyFirst;
    else if (policy=="randomly")
        m_sequencePolicy=kRandomly;
    else if (policy=="distance2origin")
        m_sequencePolicy=kSpatially;
    else if (policy=="revanC_CSR")
        m_sequencePolicy=kRevanC_CSR;
    else {
        if (policy!="singlesTime")
            G4cout<<"WARNING : policy not recognized, using default : singlesTime\n";
    m_sequencePolicy=kSinglesTime;
    }
}
//------------------------------------------------------------------------------------------------------
