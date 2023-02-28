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

	G4DigiManager *fDM = G4DigiManager::GetDMpointer();
	GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();
	//digitizerMgr->ShowSummary();

	std::string const &str = collectionName;
	std::vector<std::string> out;
	const char delim ='_';

	size_t start;
	size_t end = 0;

	GateClockDependent* module = digitizerMgr->FindElement(collectionName);


	while ((start = collectionName.find_first_not_of(delim, end)) != std::string::npos)
	{
		end = str.find(delim, start);
		out.push_back(str.substr(start, end - start));
	}
	G4int collectionID=-1;

	if(module)
	{
	    //G4cout<<module->GetObjectName()<<G4endl;
		if ( G4StrUtil::contains(module->GetObjectName(), "SinglesDigitizer"))
		{
			if (out.size()>=2)
			{

			GateSinglesDigitizer* digitizer = digitizerMgr->FindDigitizer(collectionName);
			G4int lastDCID=digitizer->m_outputDigiCollectionID;
			collectionID = lastDCID;

			}
		}
		else
		{
			collectionID = fDM->GetDigiCollectionID(collectionName);

		}
	}
	else
	{
		G4String modifiedCollectionName=out[2]+"/"+out[0]+"_"+out[1];
		collectionID = fDM->GetDigiCollectionID(modifiedCollectionName);
	}



	//std::cout << collectionName<<" "<< collectionID << std::endl;
	return collectionID;


}
