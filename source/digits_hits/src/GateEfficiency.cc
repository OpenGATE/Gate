
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateEfficiency

 	 ex-GateEfficiency and GateLocalEfficiency

  This module apples the efficiency as a function of energy.
  It uses GateVDistribution class to define either analytic
  function or list of values read from a file.


  Added to GND in 2022 by olga.kochebina@cea.fr
  Previous authors are unknown
*/

#include "GateEfficiency.hh"
#include "GateEfficiencyMessenger.hh"

#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"
#include "GateVDistribution.hh"
#include "GateSystemListManager.hh"


#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"



GateEfficiency::GateEfficiency(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_mode("crystal"),
   m_uniqueEff(-1),
   m_enabled(),
   m_efficiency_distr(0),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer),
   m_firstPass(true)
 {
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
	m_Messenger = new GateEfficiencyMessenger(this);
}


GateEfficiency::~GateEfficiency()
{
  delete m_Messenger;

}


void GateEfficiency::Digitize()
{
	//G4cout<<"GateEfficiency::Digitize "<<G4endl;
	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();



	GateDigiCollection* IDC = 0;
	IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

	GateDigi* inputDigi;


/*
	 if (nVerboseLevel==1)
			    {
			    	G4cout << "[ GateEfficiency::Digitize]: returning output digi-list with " << m_OutputDigiCollection->entries() << " entries\n");
			    	for (long unsigned int k=0; k<m_OutputDigiCollection->entries();k++)
			    		G4cout << *(*IDC)[k] << Gateendl;
			    		G4cout << Gateendl;
			    }
	*/

	 GateVSystem* system = GateSystemListManager::GetInstance()->GetSystem(0);

	 if (!system){
	       GateError("[GateEfficiency::Digitize] Problem : no system defined\n");
	      return ;
	   }

	 if (m_uniqueEff>0 && m_efficiency_distr)
	 {
		 GateError("[GateEfficiency::Digitize] Efficiency is ambiguous! Please use "
				 "/setUniqueEfficiency to choose one value or /setEfficiency to choose distribution \n");
	 }
	 if (!m_efficiency_distr && m_uniqueEff<0)
	 {
		GateError("[GateEfficiency::Digitize] No efficiency is selected! Please use "
				"/setUniqueEfficiency to choose one value or /setEfficiency to choose distribution \n");
	 }


	 if (m_mode=="crystal")
	 {
		 if (m_firstPass && m_enabled.empty()&& m_efficiency_distr)  {
			 m_firstPass=false;
			 ComputeSizes();
		 }
	 }

  if (IDC)
     {
	  G4int n_digi = IDC->entries();
	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];
		  m_outputDigi = new GateDigi(*inputDigi);

		  /* if (!m_efficiency_distr){
			  m_OutputDigiCollection->insert(m_outputDigi);
			  continue; // ???
		    }
		  */
		  G4double eff=1;
		  if (m_uniqueEff>0)
			  eff=m_uniqueEff;
		  else
		  {
			G4String UnitX = m_efficiency_distr->GetUnitX();

			if (m_mode=="energy")
				{
				if(UnitX=="keV")
					eff = m_efficiency_distr->Value(inputDigi->GetEnergy()*keV);
				else if(UnitX=="MeV")
					eff = m_efficiency_distr->Value(inputDigi->GetEnergy()*MeV);
				else
					GateError("[GateEfficiency::Digitizer] The default units of energy is keV or MeV. Please, use it too! "
							"If you need other unit of energy please contact OpenGATE developers. \n");
				}
			else
				{
					if (m_efficiency_distr)
					{
					size_t ligne;
					   ligne = system->ComputeIdFromVolID(inputDigi->GetOutputVolumeID(),m_enabled);
					   eff = m_efficiency_distr->Value(ligne);
					}
				}

		   if(eff>1)
			   GateError("[GateEfficiency::Digitize] Efficiency value is > 1.0 !!! \n");
		  }

		  if (CLHEP::RandFlat::shoot(0.,1.) < eff)
		   {
			   m_OutputDigiCollection->insert(m_outputDigi);
		   }


	  }
  }
  else
    {
  	  if (nVerboseLevel>1)
  	  	G4cout << "[GateEfficiency::Digitize]: input digi collection is null -> nothing to do\n";
  	    return;
    }
  StoreDigiCollection(m_OutputDigiCollection);

}

void GateEfficiency::SetLevel(size_t i,G4bool val)
{

   GateVSystem* system;
   size_t depth=0;
   system= GateSystemListManager::GetInstance()->GetSystem(0);
   if (!system){
	   GateError("[GateEfficiency::SetMode] Problem : no system defined\n ");
	   return;
   }
   depth = system->GetTreeDepth();

   if (m_enabled.size() != depth)
   	   {
	   if (nVerboseLevel>1)
	   	   {
		   G4cout<<"[GateEfficiency::SetMode] Enabling vector size modified from "<<m_enabled.size() <<" to "<<depth<<" and set all entries to 0\n";
	   	   }

	   	   m_enabled.resize(depth);

	   	   for (std::vector<G4bool>::iterator itr=m_enabled.begin();itr!=m_enabled.end();itr++)
    	  	  *itr=false;
   	   }

   if (i<m_enabled.size())
   	   {
	   m_enabled[i]=val;
   	   }
   else
   	   {
   		GateError("[GateEfficiency::SetMode] WARNING : index outside limits ("
   				   <<i<<">"<<m_enabled.size()<<")\n");
   	   }

}

void GateEfficiency::ComputeSizes()
{
	size_t depth=0;
    size_t totSize=0;
    GateVSystem* system;

    system= GateSystemListManager::GetInstance()->GetSystem(0);
    if (!system)
    {
    	GateError("[GateEfficiency::ComputeSizes] Problem : no system defined\n");
    	return;
     }

    depth = system->GetTreeDepth();
      if (m_enabled.size() != depth)
      	  {

    	  if (nVerboseLevel>1)
    	  	   	   {
    		  	  	  G4cout<<"[GateEfficiency::ComputeSizes] Warning : enabling vector size modified (from "<<m_enabled.size()
    	  	   	       				   <<" to "<<depth<<") and set all entries to 0\n";
    	  	   	   }
    	  	  m_enabled.resize(depth);
    	  	  for (std::vector<G4bool>::iterator itr=m_enabled.begin();itr!=m_enabled.end();itr++)
    	  		  	  *itr=false;
      	  }
      totSize = system->ComputeNofSubCrystalsAtLevel(0,m_enabled);


   if (m_enabled.size() != depth)
   	   {
	    GateError("[GateEfficiency::ComputeSizes] Warning : enabling vector size modified (from "<<m_enabled.size()
	    		<<" to "<<depth<<") and set all entries to 0\n");
	   m_enabled.resize(depth);
	   for (std::vector<G4bool>::iterator itr=m_enabled.begin();itr!=m_enabled.end();itr++)
		   	   *itr=false;
   	   }

   if (m_efficiency_distr->MaxX() < totSize-1)
   	   {
	    GateError("[GateEfficiency::ComputeSizes] Warning : efficiency table size's wrong ("<<m_efficiency_distr->MaxX()
	    		<<" instead of "<<totSize<<") disabling efficiency (all set to 1)\n");
	   m_efficiency_distr=0;
   	   }

}


void GateEfficiency::DescribeMyself(size_t indent )
{
	  G4cout << GateTools::Indent(indent) << "Tabular Efficiency "<< Gateendl;
}
