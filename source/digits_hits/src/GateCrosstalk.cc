
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateCrosstalk

  This class is not used by GATE !
  The purpose of this class is to help to create new users digitizer module(DM).

   - Create your DM by coping this class and GateDummyDigitizerMessenger class for your DM messenger
   - Places to change are marked with // ****** comment and called with "dummy" names
   - Include your module to GateSinglesDigitizerMessenger in the method DoInsertion(..)

	If you adapting some already exiting class from Old Gate Digitizer here is some of the tips
	- Digitize () is a fusion of GateVdigiProcessor::ProcessdigiList and GateXXX::Digitize
	- digi --> Digi
	- outputdigiList --> OutputDigiCollectionVector
	- inputDigi-->inputDigi
	- outputdigi --> m_outputDigi
	- how to adapt iterators check GateAdder class

  To create new Digitizer Module (DM), please, follow the steps:
  1) Copy .cc and .hh of GateCrosstalk, GateCrosstalkMessenger to GateYourNewDigitizerModule and GateYourNewDigitizerModuleMessenger
  2) Replace in these new files : Crosstalk -> YourNewDigitizerModule
  3) Compile 1st time (so not forget to redo ccmake to)
  4) Adapt GateYourNewDigitizerModuleMessenger.cc
  5) Adapt GateYourNewDigitizerModuleMessenger.hh (!!!! DO NOT FORGET TO WRITE A SHORT EXPLANATION ON WHAT DOES YOUR DM !!!!)
  6) Adapt GateYourNewDigitizerModule.hh
  7) Adapt GateYourNewDigitizerModule.cc
  	  - Change the names (x2) of YourDigitizerModule in the constructor of GateYourNewDigitizerModule.cc:
   	   	   	  line:
   	   	   	  :GateVDigitizerModule("Dummy","digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/dummy",digitizer,digitizer->GetSD()),

	  - !!!! DO NOT FORGET TO WRITE A SHORT EXPLANATION ON WHAT DOES YOUR DM !!!!
	  - Comment everything inside Digitize() method
   8) Compile 2ed time
   9) In case of adaptation of old existing class:
   	  - Copy everything inside Digitize() and ProcessdigiList() from old module to Digitize() (places where to copy are indicated in this class)
   	  - Replace:
   	  	  inputDigi -> inputDigi
	      outputdigi -> m_outputDigi + correct the first declaration (as in this Dummy module)
	      m_OutputDigiCollection->insert(outputdigi) ->  m_OutputDigiCollection->insert(m_outputDigi);
	10) Add YourDigitizerModule to GateSinglesDigitizer.cc
			- #include "YourDigitizerModule.hh"
			- in DumpMap() method in
				static G4String theList = " ...."
			- in DoInsertion() :
				  else if (childTypeName=="yourDM")
     	 	 	 	 {
   	  	  	  	  	  newDM = new GateYourDigitizerModule(m_digitizer);
   	  	  	  	  	  m_digitizer->AddNewModule(newDM);
     	 	 	 	 }
	11) Compile 3ed time and execute

	*/

#include "GateCrosstalk.hh"
#include "GateCrosstalkMessenger.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"
#include "GateObjectStore.hh"
#include "GateDetectorConstruction.hh"
#include "GateArrayParamsFinder.hh"


#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"




// Static pointer to the GateCrosstalk singleton
GateCrosstalk* GateCrosstalk::theGateCrosstalk=0;

/*    	This function allows to retrieve the current instance of the GateCrosstalk singleton
      	If the GateCrosstalk already exists, GetInstance only returns a pointer to this singleton.
	If this singleton does not exist yet, GetInstance creates it by calling the private
	GateCrosstalk constructor
*/
GateCrosstalk* GateCrosstalk::GetInstance(GateSinglesDigitizer* itsChain,
					  const G4String& itsName, G4double itsEdgesFraction,
					  G4double itsCornersFraction)
{
  if (!theGateCrosstalk)
    if (itsChain)
      theGateCrosstalk = new GateCrosstalk(itsChain, itsName, itsEdgesFraction, itsCornersFraction);
  return theGateCrosstalk;
}




GateCrosstalk::GateCrosstalk(GateSinglesDigitizer *digitizer, G4String name, G4double itsEdgesFraction, G4double itsCornersFraction)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_edgesCrosstalkFraction(itsEdgesFraction),
   m_cornersCrosstalkFraction(itsCornersFraction),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
	m_Messenger = new GateCrosstalkMessenger(this);
	m_testVolume = 0;
	CheckVolumeName(m_digitizer->GetSD()->GetName());
}


GateCrosstalk::~GateCrosstalk()
{
	 delete m_messenger;
	 delete ArrayFinder;

}


void GateCrosstalk::CheckVolumeName(G4String val)
{
  //Retrieve the inserter store to check if the volume name is valid
  GateObjectStore* m_store = GateObjectStore::GetInstance();
  if (m_store->FindCreator(val)) {
    m_volume = val;
    //Find the array params
    ArrayFinder = new GateArrayParamsFinder(m_store->FindCreator(val),
						 m_nbX, m_nbY, m_nbZ);
    m_testVolume = 1;
  }
  else {
    G4cout << "Wrong Volume Name\n";
  }
}



void GateCrosstalk::Digitize()
{
	  if(!m_testVolume)
	    {
	      G4cerr << 	Gateendl << "[GateCrosstalk::Digitize]:\n"
		     <<   "Sorry, but you don't have choosen any volume !\n";

				G4String msg = "You must choose a volume for Crosstalk, e.g. crystal:\n"
	      "\t/gate/digitizer/Singles/Crosstalk/chooseCrosstalkVolume VOLUME NAME\n"
	      "or disable the Crosstalk using:\n"
	      "\t/gate/digitizer/Singles/Crosstalk/disable\n";

				G4Exception( "GateCrosstalk::Digitize", "Digitize", FatalException, msg );
	    }



	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();



	GateDigiCollection* IDC = 0;
	IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

	GateDigi* inputDigi;

	std::vector< GateDigi* >* OutputDigiCollectionVector = m_OutputDigiCollection->GetVector ();
	std::vector<GateDigi*>::iterator iter;


	/* if (nVerboseLevel==1)
			    {
			    	G4cout << "[ GateCrosstalk::Digitize]: returning output digi-list with " << m_OutputDigiCollection->entries() << " entries\n";
			    	for (size_t k=0; k<m_OutputDigiCollection->entries();k++)
			    		G4cout << *(*IDC)[k] << Gateendl;
			    		G4cout << Gateendl;
			    }
	 */

  if (IDC)
     {
	  G4int n_digi = IDC->entries();

	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];

		  //Find the digi position in the array
		    m_depth = (size_t)(inputDigi->GetVolumeID().GetCreatorDepth(m_volume));
		    ArrayFinder->FindInputPulseParams(inputDigi->GetVolumeID().GetCopyNo(m_depth), m_i, m_j, m_k);

		    //Numbers of edge and corner neighbors for the digis
		    G4int countE = 0;
		    G4int countC = 0;

		    // Find the possible neighbors
		    if (m_edgesCrosstalkFraction != 0) {
		      if (m_i != 0) {
		        m_OutputDigiCollection->insert(CreateDigi(m_edgesCrosstalkFraction, inputDigi, m_i - 1, m_j, m_k));
		        countE++;
		      }
		      if (m_i != m_nbX - 1) {
		        m_OutputDigiCollection->insert(CreateDigi(m_edgesCrosstalkFraction, inputDigi, m_i + 1, m_j, m_k));
		        countE++;
		      }
		      if (m_j != 0) {
		        m_OutputDigiCollection->insert(CreateDigi(m_edgesCrosstalkFraction, inputDigi, m_i, m_j - 1, m_k));
		        countE++;
		      }
		      if (m_j != m_nbY - 1) {
		        m_OutputDigiCollection->insert(CreateDigi(m_edgesCrosstalkFraction, inputDigi, m_i, m_j + 1, m_k));
		        countE++;
		      }
		      if (m_k != 0) {
		        m_OutputDigiCollection->insert(CreateDigi(m_edgesCrosstalkFraction, inputDigi, m_i, m_j, m_k - 1));
		        countE++;
		      }
		      if (m_k != m_nbZ - 1) {
		        m_OutputDigiCollection->insert(CreateDigi(m_edgesCrosstalkFraction, inputDigi, m_i, m_j, m_k + 1));
		        countE++;
		      }
		    }

		    if (m_cornersCrosstalkFraction != 0) {
		      if ((m_i != 0) & (m_j != 0)) {
		        m_OutputDigiCollection->insert(CreateDigi(m_cornersCrosstalkFraction, inputDigi, m_i - 1, m_j - 1, m_k));
		        countC++;
		      }
		      if ((m_i != 0) & (m_j != m_nbY - 1)) {
		        m_OutputDigiCollection->insert(CreateDigi(m_cornersCrosstalkFraction, inputDigi, m_i - 1, m_j + 1, m_k));
		        countC++;
		      }
		      if ((m_i != 0) & (m_k != 0)) {
		        m_OutputDigiCollection->insert(CreateDigi(m_cornersCrosstalkFraction, inputDigi, m_i - 1, m_j, m_k - 1));
		        countC++;
		      }
		      if ((m_i != 0) & (m_k != m_nbZ - 1)) {
		        m_OutputDigiCollection->insert(CreateDigi(m_cornersCrosstalkFraction, inputDigi, m_i - 1, m_j, m_k + 1));
		        countC++;
		      }
		      if ((m_i != m_nbX - 1) & (m_j != 0)) {
		        m_OutputDigiCollection->insert(CreateDigi(m_cornersCrosstalkFraction, inputDigi, m_i + 1, m_j - 1, m_k));
		        countC++;
		      }
		      if ((m_i != m_nbX - 1) & (m_j != m_nbY - 1)) {
		        m_OutputDigiCollection->insert(CreateDigi(m_cornersCrosstalkFraction, inputDigi, m_i + 1, m_j + 1, m_k));
		        countC++;
		      }
		      if ((m_i != m_nbX - 1) & (m_k != 0)) {
		        m_OutputDigiCollection->insert(CreateDigi(m_cornersCrosstalkFraction, inputDigi, m_i + 1, m_j, m_k - 1));
		        countC++;
		      }
		      if ((m_i != m_nbX - 1) & (m_k != m_nbZ - 1)) {
		        m_OutputDigiCollection->insert(CreateDigi(m_cornersCrosstalkFraction, inputDigi, m_i + 1, m_j, m_k + 1));
		        countC++;
		      }
		      if ((m_j != 0) & (m_k != 0)) {
		        m_OutputDigiCollection->insert(CreateDigi(m_cornersCrosstalkFraction, inputDigi, m_i, m_j - 1, m_k - 1));
		        countC++;
		      }
		      if ((m_j != 0) & (m_k != m_nbZ - 1)) {
		        m_OutputDigiCollection->insert(CreateDigi(m_cornersCrosstalkFraction, inputDigi, m_i, m_j - 1, m_k + 1));
		        countC++;
		      }

		      if ((m_j != m_nbY - 1) & (m_k != 0)) {
		        m_OutputDigiCollection->insert(CreateDigi(m_cornersCrosstalkFraction, inputDigi, m_i, m_j + 1, m_k - 1));
		        countC++;
		      }
		      if ((m_j != m_nbY - 1) & (m_k != m_nbZ - 1)) {
		        m_OutputDigiCollection->insert(CreateDigi(m_cornersCrosstalkFraction, inputDigi, m_i, m_j + 1, m_k + 1));
		        countC++;
		      }
		    }

		    // Check if the energy of neighbors is not higher than the energy of the incident digi
		    G4double energytot = inputDigi->GetEnergy()*((countE*m_edgesCrosstalkFraction)+(countC*m_cornersCrosstalkFraction));

		    if(energytot>=inputDigi->GetEnergy())
		      {
		        G4cerr << 	Gateendl << "[GateCrosstalk::Digitize]:\n"
		  	     <<   "Sorry, but you have too much energy !\n";

		  			G4String msg = "You must change your fractions of energy for the close crystals :\n"
		        "\t/gate/digitizer/Singles/Crosstalk/setSidesFraction NUMBER\n"
		        "\t/gate/digitizer/Singles/Crosstalk/setCornersFraction NUMBER\n"
		        "or disable the Crosstalk using:\n"
		        "\t/gate/digitizer/Singles/Crosstalk/disable\n";
		  			G4Exception( "GateCrosstalk::Digitize", "Digitize", FatalException,msg);
		      }


		    // Add the incident digi in the digi list with less energy
		    m_outputDigi = new GateDigi(*inputDigi);

		    m_XtalkpCent = (1-(4*m_edgesCrosstalkFraction+4*m_cornersCrosstalkFraction));
		    m_outputDigi->SetEnergy((inputDigi->GetEnergy())*m_XtalkpCent);
		    m_OutputDigiCollection->insert(m_outputDigi);

		    if (nVerboseLevel>1)
		      G4cout << "the input digi created " << countE+countC << " digis around it"
		  	   << Gateendl;

	  } //loop  over input digits
    } //IDC
  else
    {
  	  if (nVerboseLevel>1)
  	  	G4cout << "[GateCrosstalk::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;
    }
  StoreDigiCollection(m_OutputDigiCollection);

}


GateVolumeID GateCrosstalk::CreateVolumeID(const GateVolumeID* aVolumeID, G4int i, G4int j, G4int k)
{
  GateVolumeID aVolumeIDOut;
  for (size_t n = 0; n < aVolumeID->size(); n++)
    if (n != m_depth)
      aVolumeIDOut.push_back(GateVolumeSelector(aVolumeID->GetVolume(n)));
    else {
      GateVVolume* anInserter =  aVolumeID->GetCreator(m_depth);
      G4VPhysicalVolume* aVolume = anInserter->GetPhysicalVolume(i + m_nbX * j + m_nbX * m_nbY * k);
      aVolumeIDOut.push_back(GateVolumeSelector(aVolume));
    }
  return aVolumeIDOut;
}

GateOutputVolumeID GateCrosstalk::CreateOutputVolumeID(const GateVolumeID aVolumeID)
{
	//GateDetectorConstruction* aDetectorConstruction = GateDetectorConstruction::GetGateDetectorConstruction();
	//GateOutputVolumeID anOutputVolumeID = aDetectorConstruction->GetCrystalSD()->GetSystem()->ComputeOutputVolumeID(aVolumeID);

	GateOutputVolumeID anOutputVolumeID = m_digitizer->GetSystem()->ComputeOutputVolumeID(aVolumeID);

  return anOutputVolumeID;
}

GateDigi* GateCrosstalk::CreateDigi(G4double val, const GateDigi* digi, G4int i, G4int j, G4int k)
{
  GateDigi* adigi = new GateDigi(digi);

  adigi->SetLocalPos(G4ThreeVector(0,0,0));
  adigi->SetVolumeID(CreateVolumeID(&digi->GetVolumeID(), i, j, k));
  adigi->SetGlobalPos(adigi->GetVolumeID().MoveToAncestorVolumeFrame(adigi->GetLocalPos()));
  adigi->SetOutputVolumeID(CreateOutputVolumeID(adigi->GetVolumeID()));
  adigi->SetEnergy(digi->GetEnergy()*val);

  return adigi;
}


void GateCrosstalk::DescribeMyself(size_t indent )
{
	G4cout << GateTools::Indent(indent) << "Optical Crosstalk for " << m_volume << ":\n"
		 << GateTools::Indent(indent+1) << "fraction of energy for side crystals: " << m_edgesCrosstalkFraction << "\n"
		 << GateTools::Indent(indent+1) << "fraction of energy for corner crystals: " << m_cornersCrosstalkFraction << Gateendl;
}
