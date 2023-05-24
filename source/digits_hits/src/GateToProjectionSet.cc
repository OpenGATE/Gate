/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!

  \file GateToProjectionSet.cc

  $Log: GateToProjectionSet.cc,v $

  Bug fix v6.2: 2012/09/24   by vesna.cuplov@gmail.com
  ProjectionSet is triggered by the name of the system: GateOpticalSystem or SPECThead and works
  when USE_GATE_OPTICAL is ON or OFF.

  Revision v6.2   2012/07/09  by vesna.cuplov@gmail.com
  Implemented functions that have a link with the GateToOpticalRaw class for Optical photons which is
  used to write as an output file the result of the GateToProjectionSet module for Optical Photons.

  Revision 1.1.1.1.4.1  2011/03/10 16:32:35  henri
  Implemented multiple energy window interfile output

  Revision 1.1.1.1.4.1  2011/02/02 15:37:46  henri
  Added support for multiple energy windows

  Revision 1.5  2010/12/01 17:11:23  henri
  Various bug fixes

  Revision 1.4  2010/11/30 17:48:50  henri
  Comments

  Revision 1.3  2010/11/30 16:47:08  henri
  Class GateToProjectionSet
  Modifications in order to record more than 1 input channel, as energy window

  ***Interface Changes***
  Attribute added:
  - size_t m_energyWindowNb : the number of energy window to record

  Attributes modified:
  - G4String m_inputDataChannel -> std::vector<G4String> m_inputDataChannelList
  - G4int m_inputDataChannelID -> std::vector<G4int> m_inputDataChannelIDList

  Getters and setters modified:
  - const G4String& GetInputDataName()  -> G4String& GetInputDataName(size_t energyWindowID) const : get the input channel name by index
  - void   SetOutputDataName(const G4String& aName) -> void  SetInputDataName(const G4String& aName) : changed the name

  Getters and setters added:
  - inline size_t GetEnergyWindowNb() const
  - G4int GetInputDataID(size_t energyWindowID) const : Get the input channel ID by index
  - void  AddInputDataName(const G4String& aName) : Add a input data channel name to the list
  - std::vector<G4String> GetInputDataNameList() const
  - std::vector<G4int> GetInputDataIDList() const


  ***Implementation Changes***
  In GateToProjectionSet.hh:
  - GetTotalImageNb() now returns the number of images for all energy windows

  In GateToProjectionSet.cc:
  - Slightly changed the constructor
  - RecordBeginOfAcquisition :
  *Loop over the input data channel list to get the channel IDs and fill the m_inputDataChannelIDList vector
  *Initialize the attribute m_energyWindowNb with the size of the vector m_inputDataChannelList
  *Updated GateProjectionSet::Reset

  - RecordEndOfEvent :
  *Changed some verbose text to show which digi chain is recorded (aka energy window)
  *Loop over all energy windows in m_inputDataChannelList to store all the digis related to this event



  */

#include "GateToProjectionSet.hh"

#include "globals.hh"
#include "G4UnitsTable.hh"
#include "G4Run.hh"

#include "GateDigi.hh"
#include "GateOutputMgr.hh"
#include "GateProjectionSet.hh"
#include "GateToProjectionSetMessenger.hh"
#include "GateToInterfile.hh"
#include "GateTools.hh"
#include "GateVSystem.hh"
#include "GateApplicationMgr.hh"
#include "G4DigiManager.hh"
#include "GateDigitizerMgr.hh"
#include "GateToOpticalRaw.hh" // v. cuplov

/*
 *  GateToInterfile is used to write as an output file the result of the GateToProjectionSet module.
 *  This 2 classes are strickly working together.
 *  All macro commands (inherited from GateVOutputModule) of the GateToInterfileMessenger are overloaded to
 *  have no action at all. The enable and disable command, verbose and setFileName are now managed by
 *  GateToProjectionSet. The describe command of GateToInterfile will do nothing. The enable and disable
 *  commands of the GateToProjectionSet class will also enable or disable the GateToProjectionSet module
 *  (the same for the verbose level).
 */

/*  by vesna.cuplov@gmail.com
 *  GateToOpticalRaw is used to write as a binary output file the result of the GateToProjectionSet module for
 *  optical photons.
 */

// Public constructor (creates an empty, uninitialised, project set)
GateToProjectionSet::GateToProjectionSet(const G4String& name,
                                         GateOutputMgr* outputMgr,
                                         GateVSystem* itsSystem,
                                         DigiMode digiMode) :
  GateVOutputModule(name, outputMgr, digiMode),
  m_projectionSet(0),
  m_energyWindowNb(0),
  m_projNb(0),
  m_headNb(0),
  m_orbitingStep(0.),
  m_headAngularPitch(0.),
  m_projectionPlane("Unknown"),
  m_coordX(1),
  m_coordY(2),
  m_studyDuration(0.),
  m_system(itsSystem)
{

  m_isEnabled = false; // Keep this flag false: all output are disabled by default
  m_projectionSet = new GateProjectionSet();
  //OK GND 2022
  //m_inputDataChannelList.push_back("Singles");
  m_messenger = new GateToProjectionSetMessenger(this);

  SetVerboseLevel(0);
}

GateToProjectionSet::~GateToProjectionSet()
{
  delete m_projectionSet;
}

const G4String& GateToProjectionSet::GiveNameOfFile()
{
  m_noFileName = "  "; // 2 spaces for output module with no fileName
  return m_noFileName;
}

// ======================================================================================================
// Functions for messenger commands that have a link with the GateToInterfile or GateToOpticalRaw class
void GateToProjectionSet::SetOutputFileName(const G4String& aName)
{
  // v. cuplov -- GateToOpticalRaw for optical photons
  if (m_system->GetName() == "systems/OpticalSystem") // v. cuplov -- GateToOpticalRaw for optical photons
    {
      GateToOpticalRaw* opticalrawModule = dynamic_cast<GateToOpticalRaw*>(GateOutputMgr::GetInstance()->GetModule("opticalraw"));
      if (!opticalrawModule)
        G4Exception("GateToProjectionSet::SetOutputFileName",
                    "SetOutputFileName",
                    FatalException,
                    "No GateToOpticalRaw module has been constructed, so no output can be possible with GateToProjectionSet");
      opticalrawModule->SetFileName(aName); // It is the GateToOpticalRaw module that manages the output file
    }
  else
    {
      GateToInterfile* interfileModule = dynamic_cast<GateToInterfile*>(GateOutputMgr::GetInstance()->GetModule("interfile"));
      if (!interfileModule)
        G4Exception("GateToProjectionSet::SetOutputFileName",
                    "SetOutputFileName",
                    FatalException,
                    "No GateToInterfile module has been constructed, so no output can be possible with GateToProjectionSet");
      interfileModule->SetFileName(aName); // It is the GateToInterfile module that manages the output file
    }
}

void GateToProjectionSet::SetVerboseToProjectionSetAndInterfile(G4int aVerbosity)
{
  // v. cuplov -- GateToOpticalRaw for optical photons
  if (m_system->GetName() == "systems/OpticalSystem") // v. cuplov -- GateToOpticalRaw for optical photons
    {
      GateToOpticalRaw* opticalrawModule = dynamic_cast<GateToOpticalRaw*>(GateOutputMgr::GetInstance()->GetModule("opticalraw"));
      if (!opticalrawModule)
        G4Exception("GateToProjectionSet::SetVerboseToProjectionSetAndInterfile",
                    "SetVerboseToProjectionSetAndInterfile",
                    FatalException,
                    "No GateToOpticalRaw module has been constructed, so no output can be possible with GateToProjectionSet");
      opticalrawModule->SetVerboseLevel(aVerbosity); // We update the verbosity level for the GateToOpticalRaw module
      SetVerboseLevel(aVerbosity); // We update the verbosity level for the GateToProjectionSet module
    }
  else
    {
      GateToInterfile* interfileModule = dynamic_cast<GateToInterfile*>(GateOutputMgr::GetInstance()->GetModule("interfile"));
      if (!interfileModule)
        G4Exception("GateToProjectionSet::SetVerboseToProjectionSetAndInterfile",
                    "SetVerboseToProjectionSetAndInterfile",
                    FatalException,
                    "No GateToInterfile module has been constructed, so no output can be possible with GateToProjectionSet");
      interfileModule->SetVerboseLevel(aVerbosity); // We update the verbosity level for the GateToInterfile module
      SetVerboseLevel(aVerbosity); // We update the verbosity level for the GateToProjectionSet module
    }
}

void GateToProjectionSet::SendDescribeToProjectionSetAndInterfile()
{
  // v. cuplov -- GateToOpticalRaw for optical photons
  if (m_system->GetName() == "systems/OpticalSystem") // v. cuplov -- GateToOpticalRaw for optical photons
    {
      GateToOpticalRaw* opticalrawModule = dynamic_cast<GateToOpticalRaw*>(GateOutputMgr::GetInstance()->GetModule("opticalraw"));
      if (!opticalrawModule)
        G4Exception("GateToProjectionSet::SendDescribeToProjectionSetAndInterfile",
                    "SendDescribeToProjectionSetAndInterfile",
                    FatalException,
                    "No GateToOpticalRaw module has been constructed, so no output can be possible with GateToProjectionSet");
      opticalrawModule->Describe(); // The GateToOpticalRaw module describes itself
      Describe(); // The GateToProjectionSet module describes itself
    }
  else
    {
      GateToInterfile* interfileModule = dynamic_cast<GateToInterfile*>(GateOutputMgr::GetInstance()->GetModule("interfile"));
      if (!interfileModule)
        G4Exception("GateToProjectionSet::SendDescribeToProjectionSetAndInterfile",
                    "SendDescribeToProjectionSetAndInterfile",
                    FatalException,
                    "No GateToInterfile module has been constructed, so no output can be possible with GateToProjectionSet");
      interfileModule->Describe(); // The GateToInterfile module describes itself
      Describe(); // The GateToProjectionSet module describes itself
    }
}

void GateToProjectionSet::SetEnableToProjectionSetAndInterfile()
{
  // v. cuplov -- GateToOpticalRaw for optical photons
  if (m_system->GetName() == "systems/OpticalSystem")
    {
      GateToOpticalRaw* opticalrawModule = dynamic_cast<GateToOpticalRaw*>(GateOutputMgr::GetInstance()->GetModule("opticalraw"));
      if (!opticalrawModule)
        G4Exception("GateToProjectionSet::SetEnableToProjectionSetAndInterfile",
                    "SetEnableToProjectionSetAndInterfile",
                    FatalException,
                    "No GateToOpticalRaw module has been constructed, so no output can be possible with GateToProjectionSet");
      opticalrawModule->Enable(true); // We enable the GateToOpticalRaw module
      Enable(true); // We enable the GateToProjectionSet module
    }
  else
    {
      GateToInterfile* interfileModule = dynamic_cast<GateToInterfile*>(GateOutputMgr::GetInstance()->GetModule("interfile"));
      if (!interfileModule)
        G4Exception("GateToProjectionSet::SetEnableToProjectionSetAndInterfile",
                    "SetEnableToProjectionSetAndInterfile",
                    FatalException,
                    "No GateToInterfile module has been constructed, so no output can be possible with GateToProjectionSet");
      interfileModule->Enable(true); // We enable the GateToInterfile module
      Enable(true); // We enable the GateToProjectionSet module
    }
}

void GateToProjectionSet::SetDisableToProjectionSetAndInterfile()
{
  // v. cuplov -- GateToOpticalRaw for optical photons
  if (m_system->GetName() == "systems/OpticalSystem")
    {
      GateToOpticalRaw* opticalrawModule = dynamic_cast<GateToOpticalRaw*>(GateOutputMgr::GetInstance()->GetModule("opticalraw"));
      if (!opticalrawModule)
        G4Exception("GateToProjectionSet::SetDisableToProjectionSetAndInterfile",
                    "SetDisableToProjectionSetAndInterfile",
                    FatalException,
                    "No GateToOpticalRaw module has been constructed, so no output can be possible with GateToProjectionSet");
      opticalrawModule->Enable(false); // We disable the GateToOpticalRaw module
      Enable(false); // We disable the GateToProjectionSet module
    }
  else
    {
      GateToInterfile* interfileModule = dynamic_cast<GateToInterfile*>(GateOutputMgr::GetInstance()->GetModule("interfile"));
      if (!interfileModule)
        G4Exception("GateToProjectionSet::SetDisableToProjectionSetAndInterfile",
                    "SetDisableToProjectionSetAndInterfile",
                    FatalException,
                    "No GateToInterfile module has been constructed, so no output can be possible with GateToProjectionSet");
      interfileModule->Enable(false); // We disable the GateToInterfile module
      Enable(false); // We disable the GateToProjectionSet module
    }
}
// End of functions for messenger commands that have a link with the GateToInterfile or GateToOpticalRaw class
// ======================================================================================================

// Initialisation of the projection set
void GateToProjectionSet::RecordBeginOfAcquisition()
{
  if (nVerboseLevel > 2)
    G4cout << "entering [GateToProjectionSet::RecordBeginOfAcquisition]\n";

  // First, we check that all the parameters are valid
  if (GetPixelNbX() <= 0)
    {
      G4cerr << Gateendl<< "[GateToProjectionSet::RecordBeginOfAcquisition]:\n"
             << "Sorry, but the number of pixels along X for the projection-set (" << GetPixelNbX() << ") is invalid\n";
      G4Exception( "GateToProjectionSet::RecordBeginOfAcquisition", "RecordBeginOfAcquisition", FatalException, "You must either define this number with:\n\t/gate/output/projection/pixelNumberX NUMBER\n or disable the projection-maker using:\n\t/gate/output/projection/disable\n");
    }
  if (GetPixelNbY() <= 0)
    {
      G4cerr << Gateendl<< "[GateToProjectionSet::RecordBeginOfAcquisition]:\n"
             << "Sorry, but the number of pixels along Y for the projection-set (" << GetPixelNbY() << ") is invalid\n";
      G4Exception( "GateToProjectionSet::RecordBeginOfAcquisition", "RecordBeginOfAcquisition", FatalException, "You must either define this number with:\n\t/gate/output/projection/pixelNumberX NUMBER\n or disable the projection-maker using:\n\t/gate/output/projection/disable\n");
    }
  if (GetPixelSizeX() <= 0)
    {
      G4cerr << Gateendl<< "[GateToProjectionSet::RecordBeginOfAcquisition]:\n"
             << "Sorry, but the pixel size along X for the projection-set (" << G4BestUnit(GetPixelSizeX(),"Length") << ") is invalid\n";
      G4Exception( "GateToProjectionSet::RecordBeginOfAcquisition", "RecordBeginOfAcquisition", FatalException, "You must either define this size with:\n\t/gate/output/projection/pixelSizeX SIZE UNIT\n or disable the projection-maker using:\n\t/gate/output/projection/disable\n");
    }
  if (GetPixelSizeY() <= 0)
    {
      G4cerr << Gateendl<< "[GateToProjectionSet::RecordBeginOfAcquisition]:\n"
             << "Sorry, but the pixel size along Y for the projection-set (" << G4BestUnit(GetPixelSizeY(),"Length") << ") is invalid\n";
      G4Exception( "GateToProjectionSet::RecordBeginOfAcquisition", "RecordBeginOfAcquisition", FatalException, "You must either define this size with:\n\t/gate/output/projection/pixelSizeY SIZE UNIT\n or disable the projection-maker using:\n\t/gate/output/projection/disable\n");
    }
  if (m_projectionPlane == "Unknown")
    {
      G4cerr << Gateendl<< "[GateToProjectionSet::RecordBeginOfAcquisition]:\n"
             << "Sorry, but the pixel size along Y for the projection-set (" << G4BestUnit(GetPixelSizeY(),"Length") << ") is invalid\n";
      G4Exception( "GateToProjectionSet::RecordBeginOfAcquisition", "RecordBeginOfAcquisition", FatalException, "Sorry, but you have not defined the projection plane.\nYou must either define this plane with:\n\t/gate/output/projection/projectionPlane PLANE (XY or YZ or ZX )\n or disable the projection-maker using:\n\t/gate/output/projection/disable\n");
    }

  // Added by HDS : retrieve all the input channel IDs (energy windows)
  G4DigiManager * fDM = G4DigiManager::GetDMpointer();
  GateDigitizerMgr* theDigitizerMgr = GateDigitizerMgr::GetInstance();

  	  if(theDigitizerMgr->m_SDlist.size()>1) //TODO OK GND  adapt for multiSDs
  		  {
		  GateError("***ERROR*** Multiple Sensitive detectors approach was not yet implemented for the /projection/ output. Please, contact olga.kochebina [a] cea.fr if your need this functionality to be added. \n");

  		  }
 for (std::vector<G4String>::iterator i_inputChannelName = m_inputDataChannelList.begin();
       i_inputChannelName != m_inputDataChannelList.end(); ++i_inputChannelName)
    {
	  // OK GND 2022
	  //m_inputDataChannelIDList.push_back(fDM->GetDigiCollectionID(*i_inputChannelName));

	  if (theDigitizerMgr->m_SingleDigitizersList.size() != 0)
	  {
		  GateSinglesDigitizer* aDigitizer;
		  aDigitizer = dynamic_cast<GateSinglesDigitizer*>(theDigitizerMgr->FindSinglesDigitizer(*i_inputChannelName));
		  for (long unsigned int i =0; i<aDigitizer->m_DMlist.size(); i++)
		  {
			  m_inputDataChannelIDList.push_back(fDM->GetDigiCollectionID(aDigitizer->m_DMlist[i]->GetName()+"/"+*i_inputChannelName));
		  }
	  }
	  else
	      m_inputDataChannelIDList.push_back(fDM->GetDigiCollectionID(*i_inputChannelName));


    }

 if(m_inputDataChannelList.size()==0) //no digitizer set by user
  {
	  if(theDigitizerMgr->m_SDlist.size()==1) //TODO OK GND  adapt for multiSDs
		  {
		  m_inputDataChannelIDList.push_back(0);//fDM->GetDigiCollectionID("DigiInit/Singles_"+theDigitizerMgr->m_SDlist[0]->GetName()));
		  m_inputDataChannelList.push_back("Singles_"+theDigitizerMgr->m_SDlist[0]->GetName());

		  }
	  else
	  {
		  GateError("***ERROR*** Multiple Sensitive detectors approach was not yet implemented for the /projection/ output. Please, contact olga.kochebina [a] cea.fr if your need this functionality to be added. \n");

	  }
  }


  m_energyWindowNb = m_inputDataChannelList.size();
  // Retrieve the parameters of the experiment
  G4double timeStart = GateApplicationMgr::GetInstance()->GetTimeStart();
  G4double timeStop = GateApplicationMgr::GetInstance()->GetTimeStop();
  G4double timeStep = GateApplicationMgr::GetInstance()->GetTimeSlice();
  G4double duration = timeStop - timeStart;
  m_studyDuration = duration;

  G4double fstepNumber = duration / timeStep;
  if (fabs(fstepNumber - rint(fstepNumber)) >= 1.e-5)
    {
      G4cerr << Gateendl<< "[GateToProjectionSet::RecordBeginOfAcquisition]:\n"
             << "Sorry, but the study duration (" << G4BestUnit(duration,"Time") << ") "
             << " does not seem to be a multiple of the time-slice (" << G4BestUnit(timeStep,"Time") << ").\n";
      G4Exception( "GateToProjectionSet::RecordBeginOfAcquisition", "RecordBeginOfAcquisition", FatalException, "You must change these parameters then restart the simulation\n");
    }
  m_projNb = static_cast<size_t>(rint(fstepNumber));
  if (nVerboseLevel > 2)
    G4cout << "Number of acquisition steps: " << m_projNb << Gateendl;

  // Retrieve the parameters of the rotation
  GateSystemComponent* baseComponent = m_system->GetBaseComponent();
  G4double orbitingSpeed = baseComponent->GetOrbitingVelocity();
  m_orbitingStep = orbitingSpeed * timeStep;
  if (!m_orbitingStep)
    m_orbitingStep = 360. * deg;
  if (nVerboseLevel > 2)
    G4cout << "Rotation step: " << m_orbitingStep / degree << " deg\n";

  // Retrieve the parameters of the repeater (number of heads)
  m_headNb = baseComponent->GetAngularRepeatNumber();
  if ((size_t) baseComponent->GetGenericRepeatNumber()> m_headNb){
        if (m_headNb==1){ //No angular repeater
          m_headNb=baseComponent->GetGenericRepeatNumber();
          //Number of heads set to generic repeater number
        }
        else{
            G4Exception("GateToProjectionSet::RecordBeginOfAcquisition()",
                        "SetVerboseToProjectionSetAndInterfile",
                        FatalException,
                        "Both angular and generic repeaters were employed");
        }
    }

  m_headAngularPitch = baseComponent->GetAngularRepeatPitch();
  if (!m_headAngularPitch)
    m_headAngularPitch = 360. * deg;
  if (nVerboseLevel > 2)
    G4cout << "Angular pitch between heads: " << m_headAngularPitch / degree << " deg\n";

  // Prepare the projection set
  m_projectionSet->Reset(m_energyWindowNb, m_headNb, m_projNb);

  if (nVerboseLevel > 2)
    G4cout << "leaving [GateToProjectionSet::RecordBeginOfAcquisition]\n";
}

// We leave the projection set as it is (so that it can be stored afterwards)
// but we still have to destroy the array of projection IDs
void GateToProjectionSet::RecordEndOfAcquisition()
{
}

// Reset the projection data
void GateToProjectionSet::RecordBeginOfRun(const G4Run * r)
{
  m_projectionSet->ClearData(r->GetRunID());
}

// Update the target projections with regards to the digis acquired for this event
void GateToProjectionSet::RecordEndOfEvent(const G4Event*)
{
  if (nVerboseLevel > 2)
    G4cout << "entering [GateToProjectionSet::RecordEndOfEvent]\n";

  G4DigiManager * fDM = G4DigiManager::GetDMpointer();
  //OK GND added to have entries in optical simulations without digitizer
  GateDigitizerMgr* theDigitizerMgr = GateDigitizerMgr::GetInstance();

  if(!theDigitizerMgr->m_alreadyRun)
	  for (size_t i = 0; i<theDigitizerMgr->m_digitizerIMList.size(); i++)
	  {
		  if (nVerboseLevel > 2)
			  G4cout<<"GateToProjectionSet::RecordEndOfEvent: Running digitizer as exception for a simu without digitizer"<<G4endl;
		  theDigitizerMgr->m_digitizerIMList[i]->Digitize();
	  }
  // OK GND 2022
  const GateDigiCollection * SDC;

  for (size_t energyWindowID = 0; energyWindowID < m_energyWindowNb; energyWindowID++)
    {
      SDC = dynamic_cast<const GateDigiCollection*>(fDM->GetDigiCollection(m_inputDataChannelIDList[energyWindowID]));

      if (!SDC)
        {
          if (nVerboseLevel > 2)
            G4cout << "No digi collection for this event\n"
                   << "leaving [GateToProjectionSet::RecordEndOfEvent]\n";
          continue;
        }

      G4int n_digi = SDC->entries();
      for (G4int iDigi = 0; iDigi < n_digi; iDigi++)
        {
          G4int headID = m_system->GetMainComponentIDGND((*SDC)[iDigi]);
          G4double xProj = (*SDC)[iDigi]->GetLocalPos()[m_coordX];
          G4double yProj = (*SDC)[iDigi]->GetLocalPos()[m_coordY];
          if (nVerboseLevel >= 2)
            {
              G4cout << "[GateToProjectionSet]: Processing count on head "
                     << headID
                     << " for energy window "
                     << m_inputDataChannelList[energyWindowID]
                     << " at position "
                     << G4BestUnit((*SDC)[iDigi]->GetLocalPos(), "Length")
                     << Gateendl;
              G4cout << "Extracting projection coordinates: " << G4BestUnit(xProj,"Length") << " , " << G4BestUnit(yProj,"Length") << Gateendl;
            }
          m_projectionSet->Fill(static_cast<G4int>(energyWindowID),
                                headID,
                                xProj,
                                yProj);

        }
    }

  if (nVerboseLevel > 2)
    G4cout << "leaving [GateToProjectionSet::RecordEndOfEvent]\n";
}

/* Overload of the base-class' virtual method to print-out a description of the module

   indent: the print-out indentation (cosmetic parameter)
*/
void GateToProjectionSet::Describe(size_t indent)
{
  GateVOutputModule::Describe(indent);
  G4cout << GateTools::Indent(indent)
         << "Job:                   build a set of projections from a SPECT or OPTICAL simulation\n";
  G4cout << GateTools::Indent(indent)
         << "Is enabled?            "
         << (IsEnabled() ? "Yes" : "No")
         << Gateendl;
  G4cout << GateTools::Indent(indent)
         << "Projection plane       '"
         << m_projectionPlane
         << "'\n";
  G4cout << GateTools::Indent(indent)
         << "Number of pixels       "
         << GetPixelNbX()
         << " x "
         << GetPixelNbY()
         << Gateendl;
  G4cout << GateTools::Indent(indent)
         << "Pixel size             "
         << G4BestUnit(GetPixelSizeX(), "Length")
         << " x "
         << G4BestUnit(GetPixelSizeY(), "Length")
         << Gateendl;
  G4cout << GateTools::Indent(indent)
         << "Filled?                "
         << (m_projectionSet->GetData() ? "Yes" : "No")
         << Gateendl;
  if (GetProjectionNb())
    G4cout << GateTools::Indent(indent)
           << "Number of projections: "
           << GetProjectionNb()
           << Gateendl;
  G4cout << GateTools::Indent(indent)
         << "Attached to system:    '"
         << m_system->GetObjectName()
         << "'\n";
}

// Set the sampling plane
void GateToProjectionSet::SetProjectionPlane(const G4String& planeName)
{
  if (planeName == "XY")
    {
      m_projectionPlane = planeName;
      m_coordX = 0;
      m_coordY = 1;
    }
  else if (planeName == "YZ")
    {
      m_projectionPlane = planeName;
      m_coordX = 1;
      m_coordY = 2;
    }
  else if (planeName == "ZX")
    {
      m_projectionPlane = planeName;
      m_coordX = 2;
      m_coordY = 0;
    }
  else
    {
      G4cerr << Gateendl<< "[GateToProjectionSet::SetProjectionPlane]:\n"
             << "\tI did not recognise the plane name '" << planeName << "'\n"
             << "\tValid names are 'XY', 'YZ' and 'ZX'.\n"
             << "Setting request will be ignored!\n";
    }
}
