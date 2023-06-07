/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateVOutputModule.hh"
//#include "GateOutputModuleMessenger.hh"
#include "GateTools.hh"

#include "G4DigiManager.hh"
#include "GateDigitizerMgr.hh"

GateVOutputModule::GateVOutputModule(const G4String& name, GateOutputMgr* outputMgr,DigiMode digiMode)
  : m_outputMgr(outputMgr),
    m_name(name),
    m_digiMode(digiMode),
    m_isEnabled(false)
// !!!! By default all output modules will be disabled !!!!
// !!!! Think about it now for the derived classes !!!!
// !!!! So please do not enable by default the derived classes !!!!
// !!!! and think about using the mother variable members. !!!!
{
//    m_messenger = new GateOutputModuleMessenger(this);
}

GateVOutputModule::~GateVOutputModule()
{
//    delete m_messenger;
}

/* Virtual method to print-out a description of the module

	indent: the print-out indentation (cosmetic parameter)
*/
void GateVOutputModule::Describe(size_t indent)
{
  G4cout << Gateendl << GateTools::Indent(indent) << "Output module: '" << m_name << "'\n";
}
G4int GateVOutputModule::GetCollectionID(G4String collectionName)
{
	//G4cout<<" GateVOutputModule::GetCollectionID "<< collectionName<<G4endl;
	//GetCollectionID from collection Name:
	// construct name that corresponds to G4DigiManager
	// GetID from G4DigiManager or as a last collection in Singles or Coincidence Digitizer
	G4int collectionID=-1;

	//****
	std::string const &str = collectionName;
	std::vector<std::string> collectionNamePart;
	const char delim ='_';

	size_t start;
	size_t end = 0;


	while ((start = collectionName.find_first_not_of(delim, end)) != std::string::npos)
	{
		end = str.find(delim, start);
		collectionNamePart.push_back(str.substr(start, end - start));
	}
	//****

	G4DigiManager *fDM = G4DigiManager::GetDMpointer();


	GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();
	//digitizerMgr->ShowSummary();
	GateClockDependent* module = digitizerMgr->FindElement(collectionName);

	if(module) //Getting IDs for SinglesDigitizers and CoincidenceDigitizers
	{
		if ( G4StrUtil::contains(module->GetObjectName(), "SinglesDigitizer"))
		{
			if (collectionNamePart.size()>=2)
			{

			GateSinglesDigitizer* digitizer = digitizerMgr->FindSinglesDigitizer(collectionName);
			collectionID = digitizer->m_outputDigiCollectionID;
			}
		}
		else //case for CoinSorters and CoinDigitizers
		{
			if ( G4StrUtil::contains(module->GetObjectName(), "CoincidenceDigitizer"))
			{
				if (collectionNamePart.size()>=1)
				{

					GateCoincidenceDigitizer* digitizer = digitizerMgr->FindCoincidenceDigitizer(collectionName);
					collectionID = digitizer->m_outputDigiCollectionID;
				}
			}
			else
				collectionID = fDM->GetDigiCollectionID(collectionName);

		}
	}
	else //Getting IDs for "intermediate" digitizer modules and coin digitizer modules
	{
		G4String modifiedCollectionName;
		if(collectionNamePart.size()>2) //case for SinglesDigitizers
		{
			// modified name : adder/Singles_crystal to be found by G4DigiMan
			modifiedCollectionName=collectionNamePart[2]+"/"+collectionNamePart[0]+"_"+collectionNamePart[1];
		}
		else //case for CoincidenceDigitizers
		{
			modifiedCollectionName=collectionNamePart[1]+"/"+collectionNamePart[0];
		}
		collectionID = fDM->GetDigiCollectionID(modifiedCollectionName);
	}



	//std::cout <<" **** "<< collectionName<<" "<< collectionID << std::endl;
	return collectionID;


}
