
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateBuffer

 	\brief  GateBuffer mimics the effect of limited transfer rate

    5/12/2023 added to GND by kochebina@cea.fr

 	 \sa GateBuffer, GateBufferMessenger
*/

#include "GateBuffer.hh"
#include "GateBufferMessenger.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"
#include "GateSystemListManager.hh"
#include "GateApplicationMgr.hh"


#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"



GateBuffer::GateBuffer(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_BufferSize(1),
   m_BufferPos(1),
   m_oldClock(0),
   m_readFrequency(1),
   m_doModifyTime(false),
   m_mode(0),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
	m_Messenger = new GateBufferMessenger(this);
	SetDepth(0);
 }


GateBuffer::~GateBuffer()
{
  delete m_Messenger;

}


void GateBuffer::Digitize()
{

	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();



	GateDigiCollection* IDC = 0;
	IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

	GateDigi* inputDigi;

	std::vector< GateDigi* >* OutputDigiCollectionVector = m_OutputDigiCollection->GetVector ();
	std::vector<GateDigi*>::iterator iter;

/*
	 if (nVerboseLevel==1)
			    {
			    	G4cout << "[ GateBuffer::Digitize]: returning output digi-list with " << m_OutputDigiCollection->entries() << " entries\n";
			    	for (size_t k=0; k<m_OutputDigiCollection->entries();k++)
			    		G4cout << *(*IDC)[k] << Gateendl;
			    		G4cout << Gateendl;
			    }
	*/

	   GateVSystem* system = GateSystemListManager::GetInstance()->GetSystem(0);





  if (IDC)
     {
	  G4int n_digi = IDC->entries();

	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];

		  Buffer_t clock = (Buffer_t)( (inputDigi->GetTime()-GateApplicationMgr::GetInstance()->GetTimeStart())* m_readFrequency);
		  Buffer_t deltaClocks = (m_oldClock<clock)? clock - m_oldClock : 0;

		  size_t iBuf = system->ComputeIdFromVolID(inputDigi->GetOutputVolumeID(),m_enableList);
		  	//   m_BufferPos[iBuf] = m_BufferPos[iBuf]>deltaClocks ? m_BufferPos[iBuf]-deltaClocks : 0;
		  switch (m_mode)
		  {
		  	  case 0 : m_BufferPos[iBuf] = m_BufferPos[iBuf]>deltaClocks ? m_BufferPos[iBuf]-deltaClocks : 0; break;
		  	  case 1 : if (deltaClocks>0) m_BufferPos[iBuf]=0;break;
		  }

		  //   G4cout<<"Using Buffer "<<iBuf<<" for level1="<<inputDigi->GetComponentID(1)<< Gateendl;
		  if (m_BufferPos[iBuf]+1<=m_BufferSize)
		  {
			  m_outputDigi = new GateDigi(*inputDigi);
			  if (m_doModifyTime)
			  {
				  G4double tme = GateApplicationMgr::GetInstance()->GetTimeStart()+clock/m_readFrequency;

				  if (m_mode==1) tme += 1./m_readFrequency;
				  	  m_outputDigi->SetTime(tme);
		  	   }
			  m_OutputDigiCollection->insert(m_outputDigi);
			  m_BufferPos[iBuf]++;
		  }
		  m_oldClock = clock;


	  } //loop  over input digits
    } //IDC
  else
    {
  	  if (nVerboseLevel>1)
  	  	G4cout << "[GateBuffer::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;
    }
  StoreDigiCollection(m_OutputDigiCollection);

}

void GateBuffer::SetDepth(size_t depth)
{
    GateVSystem* system = GateSystemListManager::GetInstance()->GetSystem(0);

    m_enableList.resize(system->GetTreeDepth());

    if (depth>system->GetTreeDepth()-1)
    	depth=system->GetTreeDepth()-1;

    for (size_t i=0;i<=depth;i++)
    	m_enableList[i]=true;

    for (size_t i=depth+1;i<system->GetTreeDepth();i++)
    		m_enableList[i]=false;

    size_t nofElements = system->ComputeNofSubCrystalsAtLevel(0,m_enableList);
    m_BufferPos.resize(nofElements);

    for (size_t i=0;i<nofElements;i++)
    	m_BufferPos[i]=0;
}



void GateBuffer::DescribeMyself(size_t indent )
{
	G4cout << GateTools::Indent(indent) << "Buffer: " << G4BestUnit(m_BufferSize,"Memory size")
	         << "Read @ "<< G4BestUnit(m_readFrequency,"Frequency")<< Gateendl;
}
