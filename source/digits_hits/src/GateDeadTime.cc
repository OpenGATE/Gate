/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#include "GateDeadTime.hh"
#include "G4UnitsTable.hh"
#include "GateDeadTimeMessenger.hh"
#include "GateTools.hh"
#include "GateVolumeID.hh"
#include "GateOutputVolumeID.hh"
#include "GateDetectorConstruction.hh"
#include "GateCrystalSD.hh"
#include "GateVSystem.hh"
#include "GateObjectChildList.hh"
#include "GateVVolume.hh"
#include "GateMaps.hh"


GateDeadTime::GateDeadTime(GatePulseProcessorChain* itsChain,
                           const G4String& itsName)
  : GateVPulseProcessor(itsChain,itsName)
  , m_bufferSize(0)
  , m_bufferMode(0)
{
  m_isParalysable = false;
  m_deadTime = 0;
  m_deadTimeTable = NULL;

  m_messenger = new GateDeadTimeMessenger(this);
}




GateDeadTime::~GateDeadTime()
{
  // if(m_deadTimeTable) delete(m_deadTimeTable);
  // if(m_bufferCurrentSize) delete(m_bufferCurrentSize);
  delete m_messenger;
}




void GateDeadTime::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
  static G4int m_doneOnce = 0;      // equal to 0 before first ProcessOnPulse use, then 1

  if(!m_doneOnce)
    {

      m_doneOnce = 1;
      if(!m_testVolume)
        {
          G4cerr << 	Gateendl << "[GateDeadTime::ProcessOnePulse]:\n"
                 <<   "Sorry, but you don't have chosen any volume !\n";
        }



      if (nVerboseLevel>1)
        {
          G4cout << "first pass in dead time pulse process\n" ;
          G4cout << "deadtime set at  " << m_deadTime << " ps"<< Gateendl ;
          G4cout << "mode = " << (m_isParalysable ? "paralysable":"non-paralysable") << Gateendl ;
        }

    }
  if (!inputPulse)
    {
      if (nVerboseLevel>1)
      	G4cout << "[GateDeadTime::ProcessOnePulse]: input pulse was null -> nothing to do\n\n";
      return;
    }

  if (inputPulse->GetEnergy()==0)
    {
      if (nVerboseLevel>1)
        G4cout << "[GateDeadTime::ProcessOneHit]: energy is null for " << inputPulse << " -> pulse ignored\n\n";
      return;
    }


  // FIND THE ELEMENT ID OF PULSE
  const GateVolumeID* aVolumeID = &inputPulse->GetVolumeID();
  G4int m_generalDetId = 0; // a unique number for each detector part
                            // that depends of the depth of application
                            // of the dead time
  size_t m_depth = (size_t)(aVolumeID->GetCreatorDepth(m_volumeName));

  m_generalDetId = aVolumeID->GetCopyNo(m_depth);
  //G4cout<<"m_generalDetId initialisation = "<<m_generalDetId<< Gateendl;
  //G4cout<<"m_depth  = "<<m_depth<< Gateendl;


  /////// Bug Report - 8/6/2006 - Spencer Bowen - S.Jan ////////
  /*

    for(G4int i = 1 ; i < numberOfHigherLevels + 1; i++)
    {
    m_generalDetId += aVolumeID->GetCopyNo(m_depth-i) * numberOfComponentForLevel[i-1];


    }
  */
  G4int multFactor = 1;
  for(G4int i = 1 ; i < numberOfHigherLevels + 1; i++)
    {
      multFactor *= numberOfComponentForLevel[i-1];
      m_generalDetId += aVolumeID->GetCopyNo(m_depth-i)*multFactor;

      //G4cout<<"numberOfComponentForLevel["<<i-1<<"] = "<<numberOfComponentForLevel[i-1]<< Gateendl;
      //G4cout<<"GetCopyNo(m_depth-i) = "<<aVolumeID->GetCopyNo(m_depth-i)<< Gateendl;
      //G4cout<<"multFactor"<<multFactor<< Gateendl;
    }
  //////////////////////////////////////////////////////////////
  // FIND TIME OF PULSE
  unsigned long long int currentTime = (unsigned long long int)((inputPulse->GetTime())/picosecond);
  if (nVerboseLevel>5)
    {
      G4cout << "A new pulse is processed by dead time time : " << (inputPulse->GetTime())/picosecond
             << " =  "<< currentTime  << Gateendl  ;
      G4cout << "ID elt = " <<  m_generalDetId << Gateendl ;
      G4cout << "Rebirth time for elt " << m_generalDetId << " = " << m_deadTimeTable[m_generalDetId]<< Gateendl ;

    }


  // IS DETECTOR DEAD ?
  if (currentTime >=  m_deadTimeTable[m_generalDetId])
    {
      // NO DETECTOR IS NOT DEAD : COPY THIS PULSE TO OUTPUT PULSE LIST
      GatePulse* outputPulse = new GatePulse(*inputPulse);
      outputPulseList.push_back(outputPulse);

      //      m_deadTimeTable[m_generalDetId] = currentTime + m_deadTime;
      if (m_bufferSize>1){
      	m_bufferCurrentSize[m_generalDetId]++;
        if (m_bufferCurrentSize[m_generalDetId]==m_bufferSize){
          m_deadTimeTable[m_generalDetId] = currentTime + m_deadTime;
          m_bufferCurrentSize[m_generalDetId]=0;
        }
      } else {
      	m_deadTimeTable[m_generalDetId] = currentTime + m_deadTime;
      }
      if (nVerboseLevel>5){
        G4cout << "We have accept " << currentTime << " a pulse in element " << m_generalDetId <<
          "\trebirth time\t" << m_deadTimeTable[m_generalDetId] << Gateendl;
        G4cout << "Copied pulse to output:\n"
               << *outputPulse << Gateendl << Gateendl ;
      }
    }
  else
    {
      // YES DETECTOR IS DEAD : REMOVE PULSE
      if (nVerboseLevel>5)
        G4cout << "Removed pulse, due to dead time:\n";
      // AND IF "PARALYSABLE" DEAD TIME, MAKE THE DEATH OF DETECTOR LONGER
      if ((m_bufferSize>1) && (m_bufferMode==1)){
        if (m_bufferCurrentSize[m_generalDetId]<m_bufferSize-1) {
          m_bufferCurrentSize[m_generalDetId]++;
          outputPulseList.push_back(new GatePulse(*inputPulse));
        }
      } else {
      	if(m_isParalysable && (m_bufferSize<2)){
          m_deadTimeTable[m_generalDetId]  = currentTime + m_deadTime;
        }
      }
    }
  if (nVerboseLevel>99)
    getchar();
}



void GateDeadTime::SetDeadTimeMode(G4String val)
{
  if((val!="paralysable")&&(val!="nonparalysable"))
    G4cout << "*** GateDeadTime.cc : Wrong dead time mode : candidates are : paralysable nonparalysable\n";
  else
    m_isParalysable = (val=="paralysable");

}

void GateDeadTime::CheckVolumeName(G4String val)
{
  GateObjectStore* anInserterStore = GateObjectStore::GetInstance();


  if (anInserterStore->FindCreator(val)) {
    m_volumeName = val;

    FindLevelsParams(anInserterStore);
    m_testVolume = 1;
  }
  else {
    G4cout << "Wrong Volume Name\n";
  }
}



void GateDeadTime::FindLevelsParams(GateObjectStore*  anInserterStore)
{


  G4int numberTotalOfComponentInSystem = 0;

  GateVVolume* anInserter = anInserterStore->FindCreator(m_volumeName);
  GateVVolume* anotherInserter = anInserter; // just to buffer anInserter


  if(nVerboseLevel>1)
    G4cout << "DEAD TIME IS APPLIED ON " <<  m_volumeName << Gateendl;


  // How many levels higher than volumeName level ?
  numberOfHigherLevels = 0;
  while(anotherInserter->GetMotherList())
    {
      anotherInserter =  anotherInserter->GetMotherList()->GetCreator();
      numberOfHigherLevels ++;
    }
  //  numberOfHigherLevels--;
  anotherInserter = anInserter;
  G4cout<<"Nof Higher level "<<numberOfHigherLevels<< Gateendl;

  // How many components for each levels ?
  //  G4int* numberOfComponentForLevel = new G4int[numberOfHigherLevels];
  numberOfComponentForLevel = (G4int*)malloc(sizeof(G4int)*(numberOfHigherLevels));
  if(!numberOfComponentForLevel)
    {
      G4cout << "[GateDeadTime::FindLevelsParams]: new failed\n\n";
      return;
    }


  numberOfComponentForLevel[0] = anotherInserter->GetVolumeNumber();


  for(G4int i = 1 ; i < numberOfHigherLevels ; i++)
    {
      anotherInserter = anotherInserter->GetMotherList()->GetCreator();
      numberOfComponentForLevel[i] = anotherInserter->GetVolumeNumber();
    }

  numberTotalOfComponentInSystem = 1;
  for(G4int i2 = 0 ; i2 < numberOfHigherLevels ; i2++)
    {
      numberTotalOfComponentInSystem = numberTotalOfComponentInSystem * numberOfComponentForLevel[i2];
      if(nVerboseLevel>5)
        G4cout << "Level : " << i2 << " has "
               << numberOfComponentForLevel[i2] << " elements\n";
    }

  //G4cout<<"numberOfComponentForLevel[0]"<<numberOfComponentForLevel[0]<< Gateendl;
  //G4cout<<"numberOfComponentForLevel[numberOfHigherLevels-1]"<<numberOfComponentForLevel[numberOfHigherLevels-1]<< Gateendl;


  if(nVerboseLevel>5)
    G4cout << "total number of elements = " <<numberTotalOfComponentInSystem << Gateendl;

  // create the table of "rebirth time" (detector is dead than it rebirth)
  //unsigned long long int* m_deadTimeTable = new unsigned long long int[numberTotalOfComponentInSystem];
  m_deadTimeTable = (unsigned long long int*)malloc(sizeof(unsigned long long int)*numberTotalOfComponentInSystem);
  m_bufferCurrentSize = (G4double*)malloc(sizeof(G4double)*numberTotalOfComponentInSystem);

  //  printf("alocated for %d elements\n",numberTotalOfComponentInSystem);getchar();


  if( (!m_deadTimeTable) || (!m_bufferCurrentSize)){
    G4cout << "[GateDeadTime::FindLevelsParams]: malloc failed\n\n";
    return;
  } else {
    for(G4int i=0;i<numberTotalOfComponentInSystem;i++){
      m_deadTimeTable[i] = 0;
      m_bufferCurrentSize[i] = 0.;

    }
  }
}





void GateDeadTime::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "DeadTime: " << G4BestUnit(m_deadTime,"Time") << Gateendl;
}
