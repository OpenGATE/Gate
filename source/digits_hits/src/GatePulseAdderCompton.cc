/*----------------------
OpenGATE Collaboration

Daniel Strul <daniel.strul@iphe.unil.ch>
JB Michaud <jbmichaud@videotron.ca>

Copyright (C) 2002,2003 UNIL/IPHE, CH-1015 Lausanne
Copyright (C) 2009 Universite de Sherbrooke

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GatePulseAdderCompton.hh"

#include "G4UnitsTable.hh"

#include "GatePulseAdderComptonMessenger.hh"
#include "G4Electron.hh"
#include "GateConfiguration.h"

GatePulseAdderCompton::GatePulseAdderCompton(GatePulseProcessorChain* itsChain,
											 const G4String& itsName)
											 : GateVPulseProcessor(itsChain,itsName)
{
	m_messenger = new GatePulseAdderComptonMessenger(this);
}

GatePulseAdderCompton::~GatePulseAdderCompton()
{
	delete m_messenger;
}


//rewritten for Compton
void GatePulseAdderCompton::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
#ifdef GATE_USE_OPTICAL
	// ignore pulses based on optical photons. These can be added using the opticaladder
	if (!inputPulse->IsOptical())
#endif
	{
		//G4cout << *inputPulse << " ";
		if ( outputPulseList.empty() )
		{
			//G4cout << inputPulse->GetEventID() << " " << inputPulse->GetOutputVolumeID() << " " << inputPulse->GetEnergy() << " " << inputPulse->GetPDGEncoding() << " First" ;
			if ( inputPulse->GetPDGEncoding() == ( G4Electron::Electron()->GetPDGEncoding() ) )
			{
				//G4cout <<  " Different Discarded" << G4endl;

				if (nVerboseLevel>1)
					G4cout << "Discarded electronic pulse for volume " << inputPulse->GetVolumeID()
					<< " with no previous photonic interaction in that Volume ID" << G4endl << G4endl ;
			}
			else
			{
				PulsePushBack(inputPulse, outputPulseList);
				//G4cout << " Different Push Back" << G4endl;
			}
		}
		else
		{
			GatePulseList::reverse_iterator currentiter = outputPulseList.rbegin();

			while (1)
			{
				if ( inputPulse->GetPDGEncoding() == ( G4Electron::Electron()->GetPDGEncoding() ) )
				{
					if ( (inputPulse->GetVolumeID() == (*currentiter)->GetVolumeID()) && (inputPulse->GetEventID() == (*currentiter)->GetEventID()) )
					{
						(*currentiter)->CentroidMergeCompton(inputPulse);
						if (nVerboseLevel>1)
							G4cout << "Merged past photonic pulse for volume " << inputPulse->GetVolumeID()
							<< " with new electronic pulse of energy " << G4BestUnit(inputPulse->GetEnergy(),"Energy") <<".\n"
							<< "Resulting pulse is: " << G4endl
							<< (*currentiter) << G4endl << G4endl ;
						//G4cout << inputPulse->GetEventID() << " " << inputPulse->GetOutputVolumeID() << " " << inputPulse->GetEnergy() << " " << inputPulse->GetPDGEncoding() << " Merged" << G4endl;
						break;
					}
					else
					{
						//G4cout <<  "Increment " ;
						currentiter++;
						if (currentiter == outputPulseList.rend())
						{
							//G4cout << inputPulse->GetEventID() << " " << inputPulse->GetOutputVolumeID() << " " << inputPulse->GetEnergy() << " " << inputPulse->GetPDGEncoding() << " End of list" << G4endl;
							break;
						}
					}
				}
				else
				{
					//G4cout << inputPulse->GetEventID() << " " << inputPulse->GetOutputVolumeID() << " " << inputPulse->GetEnergy() << " " << inputPulse->GetPDGEncoding() << " Push back" << G4endl;
					PulsePushBack(inputPulse, outputPulseList);
					break;
				}

			}

		}
	}
}


//void GatePulseAdderCompton::DescribeMyself(size_t indent)
void GatePulseAdderCompton::DescribeMyself(size_t )
{
	;
}

//this is standalone only because it repeats twice in processOnePulse()
inline void GatePulseAdderCompton::PulsePushBack(const GatePulse* inputPulse, GatePulseList& outputPulseList)
{
	GatePulse* outputPulse = new GatePulse(*inputPulse);
	if (nVerboseLevel>1)
		G4cout << "Created new pulse for volume " << inputPulse->GetVolumeID() << ".\n"
		<< "Resulting pulse is: " << G4endl
		<< *outputPulse << G4endl << G4endl ;
	outputPulseList.push_back(outputPulse);
}

//this is standalone only because it repeats twice in processOnePulse()
/*inline void GatePulseAdderCompton::DifferentVolumeIDs(const GatePulse* inputPulse, GatePulseList& outputPulseList)
{
}*/

//overload base virtual function so repack can be called after processing
/*GatePulseList* GatePulseAdderCompton::ProcessPulseList(const GatePulseList *inputPulseList)
{
	GatePulseList* outputPulseList = GateVPulseProcessor::ProcessPulseList(inputPulseList);
	if (!(outputPulseList->empty()))
		repackLastVolumeID(*outputPulseList);
	return outputPulseList;
}*/

//this is the tricky non-Gate compliant part. the list must be unfolded so that photonic interactions
//can be merged
//this could be called several times on the same last volume ID
/*void GatePulseAdderCompton::repackLastVolumeID(GatePulseList& outputPulseList)
{
	const GatePulseListSizeType MinSize = 1;

	//wont run if outputlist is obviously undersized
	if (outputPulseList.size() > MinSize)
	{
		GatePulseReverseIterator currentiter = outputPulseList.rbegin();
		GatePulseReverseIterator previousiter = currentiter++;
		G4int count = 1;

		//compute how many photonic interactions there were in the last volume id
		while ( (currentiter != outputPulseList.rend()) &&
			( ((*currentiter)->GetVolumeID() == (*previousiter)->GetVolumeID() )) && ((*currentiter)->GetEventID() == (*previousiter)->GetEventID()) )
		{
			count++;
			currentiter++;
			previousiter++;
		}

		//merge all interactions inside that volume id
		GatePulseListElementReference outputListBack = outputPulseList.back();
		for (;count > 1; count--)
		{
			//create a temp storage for the last pulse
			GatePulse* tempPulse = new GatePulse((*outputListBack));
			//get rid of the list entry now unused
			outputPulseList.pop_back();
			//free memory assigned for that pulse in PulsePushBack() above
			delete outputListBack;
			//get the new proper end
			outputListBack = outputPulseList.back();
			//merge pulses
			outputListBack->CentroidMerge(tempPulse);
			//get rid of temp
			delete tempPulse;
			if (m_verboseLevel>1)
				G4cout << "Merged photonic pulse for volume " << tempPulse->GetVolumeID()
				<< " with other photonic pulse of energy " << G4BestUnit(tempPulse->GetEnergy(),"Energy") <<".\n"
				<< "Resulting pulse is: " << G4endl
				<< (*outputListBack) << G4endl << G4endl ;
		}
	}
}
*/
