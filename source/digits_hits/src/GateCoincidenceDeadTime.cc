/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCoincidenceDeadTime.hh"
#include "G4UnitsTable.hh"
#include "GateCoincidenceDeadTimeMessenger.hh"
#include "GateTools.hh"
#include "GateVolumeID.hh"
#include "GateOutputVolumeID.hh"
#include "GateVVolume.hh"
#include "GateDetectorConstruction.hh"
#include "GateCrystalSD.hh"
#include "GateVSystem.hh"
#include "GateObjectChildList.hh"
#include "GateMaps.hh"





GateCoincidenceDeadTime::GateCoincidenceDeadTime(GateCoincidencePulseProcessorChain* itsChain,
			   const G4String& itsName)
  : GateVCoincidencePulseProcessor(itsChain,itsName)
{
  m_isParalysable = false;
  m_deadTime = 0;
  m_rebirthTime=0;
  m_bufferCurrentSize=0;
  m_bufferSize=0;
  m_oldEv1=-1;
  m_oldEv2=-1;
  m_oldName="";
  m_conserveAllEvent=true;

  m_messenger = new GateCoincidenceDeadTimeMessenger(this);
}




GateCoincidenceDeadTime::~GateCoincidenceDeadTime()
{
  delete m_messenger;
}




GateCoincidencePulse* GateCoincidenceDeadTime::ProcessPulse(GateCoincidencePulse* inputPulse,G4int iPulse)
{
  if (!inputPulse || inputPulse->size()!=2) {
      if (nVerboseLevel>1)
      	G4cout << "[GateCoincidenceDeadTime::ProcessOnePulse]: input pulse was null -> nothing to do\n\n";
      return 0;
  }
  unsigned long long int  currentTime = (unsigned long long int)(inputPulse->GetTime()/picosecond);


  if (m_conserveAllEvent){
    if (iPulse==0) {
    	m_oldEv1 = (*inputPulse)[0]->GetEventID();
    	m_oldEv2 = (*inputPulse)[1]->GetEventID();
    } else {
        if (m_wasTaken && m_bufferSize>0) m_bufferCurrentSize++;
      	return m_wasTaken?new GateCoincidencePulse(*inputPulse):0;
      }
  }
  // FIND TIME OF PULSE
  if (nVerboseLevel>5){
      G4cout << "A new pulse is processed by dead time time : " << (inputPulse->GetTime())/picosecond
	     << " =  "<< currentTime  <<G4endl  ;
      G4cout << "Rebirth time is " << m_rebirthTime << G4endl ;

  }
  GateCoincidencePulse* outputPulse=0;
  // IS DETECTOR DEAD ?
  if (currentTime >=  m_rebirthTime) {
      // NO DETECTOR IS NOT DEAD : COPY THIS PULSE TO OUTPUT PULSE
      outputPulse= new GateCoincidencePulse(*inputPulse);
      if (m_bufferSize>0){
      	m_bufferCurrentSize++;
	if (m_bufferCurrentSize>=m_bufferSize){
	    m_rebirthTime = currentTime + m_deadTime;
	    m_bufferCurrentSize=0;
	}
      } else {
      	m_rebirthTime = currentTime + m_deadTime;
      }
      if (nVerboseLevel>5){
	 G4cout << "We have accept " << currentTime << " a pulse "
	   <<"\trebirth time\t" << m_rebirthTime << G4endl;
	 G4cout << "Copied pulse to output:" << G4endl
		<< *outputPulse << G4endl << G4endl ;
      }
  } else {
      // YES DETECTOR IS DEAD : MAY BE REMOVE PULSE
      if ((m_bufferSize>0) && (m_bufferMode==1)){
	if (m_bufferCurrentSize<m_bufferSize) {
      	    m_bufferCurrentSize++;
      	    outputPulse= new GateCoincidencePulse(*inputPulse);
	    if (nVerboseLevel>5){
	       G4cout << "We have accept " << currentTime << " a pulse "
		 <<"\trebirth time\t" << m_rebirthTime << G4endl;
	       G4cout << "Copied pulse to output:" << G4endl
		      << *outputPulse << G4endl << G4endl ;
	    }
            if (m_isParalysable && (m_bufferCurrentSize==m_bufferSize)){
                m_rebirthTime  = currentTime + m_deadTime;
	    }
	} else {
      	    if (nVerboseLevel>5)
	    	G4cout << "Removed pulse, due to dead time." << G4endl;
	    outputPulse=0;
	}
      } else {
      // AND IF "PARALYSABLE" DEAD TIME, MAKE THE DEATH OF DETECTOR LONGER
	 if (m_isParalysable && (m_bufferSize<1)){
	     m_rebirthTime  = currentTime + m_deadTime;
	 }
      	 if (nVerboseLevel>5)
	     G4cout << "Removed pulse, due to dead time." << G4endl;
	 outputPulse=0;
      }
  }
  if (m_conserveAllEvent && (iPulse==0)) m_wasTaken = (outputPulse!=0);
  if (nVerboseLevel>99)
    getchar();
  return outputPulse;
}



void GateCoincidenceDeadTime::SetDeadTimeMode(G4String val)
{
  if((val!="paralysable")&&(val!="nonparalysable"))
    G4cout << "*** GateCoincidenceDeadTime.cc : Wrong dead time mode : candidates are : paralysable nonparalysable" << G4endl;
  else
   m_isParalysable = (val=="paralysable");

}

void GateCoincidenceDeadTime::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "DeadTime: " << G4BestUnit(m_deadTime,"Time") << G4endl;
}
