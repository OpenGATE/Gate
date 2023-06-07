/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*! \file GateToOpticalRaw.cc
   Created on   2012/07/09  by vesna.cuplov@gmail.com
   Implemented new class GateToOpticalRaw for Optical photons: write result of the projection.
*/

#include "GateToOpticalRaw.hh"
#include "GateToOpticalRawMessenger.hh"
#include "globals.hh"
#include "G4Run.hh"

#include "GateOutputMgr.hh"
#include "GateTools.hh"
#include "GateOpticalSystem.hh"
#include "GateToProjectionSet.hh"
#include "GateProjectionSet.hh"

#include "GateDigitizerMgr.hh"



GateToOpticalRaw::GateToOpticalRaw(const G4String& name, GateOutputMgr* outputMgr,GateOpticalSystem* itsSystem,DigiMode digiMode)
  : GateVOutputModule(name,outputMgr,digiMode)
  , m_system(itsSystem)
  , m_fileName(" ") // All default output file from all output modules are set to " ".
                    // They are then checked in GateApplicationMgr::StartDAQ, using
                    // the VOutputModule pure virtual method GiveNameOfFile()
{

  m_isEnabled = false; // Keep this flag false: all output are disabled by default
  m_asciiMessenger = new GateToOpticalRawMessenger(this);

  nVerboseLevel =0;
}




GateToOpticalRaw::~GateToOpticalRaw()
{
  delete m_asciiMessenger;

  if (nVerboseLevel > 0) G4cout << "GateToOpticalRaw deleting...\n";

  if (m_headerFile.is_open())
    m_headerFile.close();

  if (m_dataFile.is_open())
    m_dataFile.close();
}


const G4String& GateToOpticalRaw::GiveNameOfFile()
{
  return m_fileName;
}

void GateToOpticalRaw::RecordBeginOfAcquisition()
{
  if (!(m_system->GetProjectionSetMaker()->IsEnabled())) return;

 // Open the header file
  m_headerFile.open((m_fileName+".hdr").c_str(),std::ios::out | std::ios::trunc);
  if (!(m_headerFile.is_open()))
	{
		G4String msg = "Could not open the header file '" + m_fileName + ".hdr'!";
      	G4Exception("GateToOpticalRaw::RecordBeginOfAcquisition", "RecordBeginOfAcquisition", FatalException,msg);
	}
  // Pre-write the header file
  WriteGeneralInfo();
  m_headerFile  << "!END OF INTERFILE :="    	      	    << Gateendl;

  // Open the data file
  m_dataFile.open((m_fileName+".bin").c_str(),std::ios::out | std::ios::trunc | std::ios::binary);
  if (!(m_dataFile.is_open()))
	{
		G4String msg = "Could not open the data file '" + m_fileName + ".bin'!";
      	G4Exception("GateToOpticalRaw::RecordBeginOfAcquisition", "RecordBeginOfAcquisition", FatalException,msg);
}
}


void GateToOpticalRaw::RecordEndOfAcquisition()
{
  if (!(m_system->GetProjectionSetMaker()->IsEnabled())) return;

  // Close the data file
  m_dataFile.close();

  // Fully rewrite the header, so as to store the maximum counts
  m_headerFile.seekp(0,std::ios::beg);
  if ( m_headerFile.bad() ) G4Exception( "GateToOpticalRaw::RecordEndOfAcquisition", "RecordEndOfAcquisition", FatalException, "Could not go to back to the beginning of the header file (file missing?)!\n");
  WriteGeneralInfo();
  m_headerFile  << "!END OF INTERFILE :="    	      	    << Gateendl;
  m_headerFile.close();


}


void GateToOpticalRaw::RecordBeginOfRun(const G4Run* )
{
  if (!(m_system->GetProjectionSetMaker()->IsEnabled())) return;
}





void GateToOpticalRaw::RecordEndOfRun(const G4Run* )
{
  if (!(m_system->GetProjectionSetMaker()->IsEnabled())) return;

  // Write the projection sets
	if (m_system->GetProjectionSetMaker()->GetProjectionSet()->GetData() != 0) {
		for (size_t energyWindowID = 0; energyWindowID < m_system->GetProjectionSetMaker()->GetEnergyWindowNb(); energyWindowID++) {
  			for (size_t headID=0 ; headID < m_system->GetProjectionSetMaker()->GetHeadNb(); headID++) {

				m_system->GetProjectionSetMaker()->GetProjectionSet()->StreamOut( m_dataFile , energyWindowID, headID );

			}
		}
	}

	else if (m_system->GetProjectionSetMaker()->GetProjectionSet()->GetARFData() != 0) {
		for (size_t headID=0 ; headID < m_system->GetProjectionSetMaker()->GetHeadNb(); headID++) {
			m_system->GetProjectionSetMaker()->GetProjectionSet()->StreamOutARFProjection( m_dataFile , headID );
		}

	} else {
		G4cerr << "[GateToOpticalRaw::RecordEndOfRun]:\n"
		<< "No data available to write to projection set.\n";
	}

}



/* Overload of the base-class' virtual method to print-out a description of the module

	indent: the print-out indentation (cosmetic parameter)
*/
void GateToOpticalRaw::Describe(size_t indent)
{
  GateVOutputModule::Describe(indent);
  G4cout << GateTools::Indent(indent) << "Job:                   write a set of projections into an raw output file\n";
  G4cout << GateTools::Indent(indent) << "Is enabled?            " << ( IsEnabled() ? "Yes" : "No") << Gateendl;
  G4cout << GateTools::Indent(indent) << "File name:             '" << m_fileName << "'\n";
  G4cout << GateTools::Indent(indent) << "Attached to system:    '" << m_system->GetObjectName() << "'\n";
}



// Write the general INTERFILE information into the header
void GateToOpticalRaw::WriteGeneralInfo()
{
  m_headerFile  << "!INTERFILE :="    	      	      	    << Gateendl
		<< "!imaging modality := "       	    << "optical imaging\n"
      	      	<< ";\n";

  m_headerFile  << "!GENERAL DATA :=" 	      	      	    << Gateendl
		<< "data description := "     	      	    << "GATE simulation\n"
 		<< "!name of data file := "    	      	    << m_fileName+".bin\n"
    	      	<< ";\n";

  GateToProjectionSet* setMaker = m_system->GetProjectionSetMaker();

  m_headerFile  << "!GENERAL IMAGE DATA :="   	      	    << Gateendl
		<< "!type of data := "     	      	    << "OPTICAL\n"
 		<< "!total number of images := "      	    << setMaker->GetTotalImageNb() << Gateendl
 	      	<< ";\n";

    // Modified by HDS : multiple energy windows support
	//------------------------------------------------------------------
  	  //OK GND 2022

  GateDigitizerMgr* theDigitizerMgr = GateDigitizerMgr::GetInstance();
  GateSinglesDigitizer* aDigitizer;

  G4String aChainName;

	// Loop over the energy windows first and then over detector heads
	for (size_t energyWindowID=0; energyWindowID < setMaker->GetEnergyWindowNb(); energyWindowID++) {

		// Get the pulse processor chain pointer for the current energy window
		aChainName = setMaker->GetInputDataName(energyWindowID);
		aDigitizer = dynamic_cast<GateSinglesDigitizer*>(theDigitizerMgr->FindSinglesDigitizer(aChainName));
				if (!aDigitizer) {
					G4cerr  << 	Gateendl << "[GateToOpticalRaw::WriteGeneralInfo]:\n"
							<< "Can't find digitizer chain '" << aChainName << "', aborting\n";
					G4Exception( "GateToOpticalRaw::WriteGeneralInfo", "WriteGeneralInfo", FatalException, "You must change this parameter then restart the simulation\n");
				}


		m_headerFile  << "!OPTICAL STUDY (general) :="        	    << Gateendl
      	   	<< "number of detector heads := "     	    << setMaker->GetHeadNb() << Gateendl
 	      	<< ";\n";

		// Write description for each head
		for (size_t headID=0; headID<setMaker->GetHeadNb() ; headID++) {

			m_headerFile  << "!number of images divided by number of energy window := "    << setMaker->GetTotalImageNb()  / setMaker->GetEnergyWindowNb() << Gateendl
				  << "projection matrix size [1] := "     	      << setMaker->GetPixelNbX() << Gateendl
				  << "projection matrix size [2] := "     	      << setMaker->GetPixelNbY() << Gateendl
				  << "projection pixel size along X-axis (cm) [1] := "     << setMaker->GetPixelSizeX()/cm << Gateendl
				  << "projection pixel size along Y-axis (cm) [2] := "     << setMaker->GetPixelSizeY()/cm << Gateendl
				  << "!number of projections := "             << setMaker->GetProjectionNb()<< Gateendl
				  << "!extent of rotation := "       	      << setMaker->GetAngularSpan()/deg << Gateendl
				  << "!time per projection (sec) := "         << setMaker->GetTimePerProjection() / second << Gateendl
				  << ";\n";
		}

	}

  m_headerFile  << ";GATE GEOMETRY :="               	  << Gateendl;

  GateVVolume *baseInserter  = m_system->GetBaseComponent()->GetCreator();
  m_headerFile  << ";Optical System x dimension (cm) := "            <<  2.* baseInserter->GetCreator()->GetHalfDimension(0)/cm << Gateendl
      	      	<< ";Optical System y dimension (cm) := "            <<  2.* baseInserter->GetCreator()->GetHalfDimension(1)/cm << Gateendl
      	      	<< ";Optical System z dimension (cm) := "            <<  2.* baseInserter->GetCreator()->GetHalfDimension(2)/cm << Gateendl
      	      	<< ";Optical System material := "                	  <<  baseInserter->GetCreator()->GetMaterialName() << Gateendl
		<< ";Optical System x translation (cm) := "      	  <<  baseInserter->GetVolumePlacement()->GetTranslation().x()/cm << Gateendl
		<< ";Optical System y translation (cm) := "      	  <<  baseInserter->GetVolumePlacement()->GetTranslation().y()/cm << Gateendl
		<< ";Optical System z translation (cm) := "      	  <<  baseInserter->GetVolumePlacement()->GetTranslation().z()/cm << Gateendl;

  GateVVolume *crystalInserter  = m_system->GetCrystalComponent()->GetCreator();
  if ( crystalInserter )
    m_headerFile  << ";"          << Gateendl
                  << ";Optical System LEVEL 1 element is crystal := "          << Gateendl
                  << ";Optical System crystal x dimension (cm) := "         <<  2.* crystalInserter->GetCreator()->GetHalfDimension(0)/cm << Gateendl
      	      	  << ";Optical System crystal y dimension (cm) := "         <<  2.* crystalInserter->GetCreator()->GetHalfDimension(1)/cm << Gateendl
      	      	  << ";Optical System crystal z dimension (cm) := "         <<  2.* crystalInserter->GetCreator()->GetHalfDimension(2)/cm << Gateendl
      	      	  << ";Optical System crystal material := "                 <<  crystalInserter->GetCreator()->GetMaterialName() << Gateendl;

  GateVVolume *pixelInserter  = m_system->GetPixelComponent()->GetCreator();
  if ( pixelInserter )
    m_headerFile  << ";"          << Gateendl
                  << ";Optical System LEVEL 2 element is pixel := "          << Gateendl
                  << ";Optical System pixel x dimension (cm) := "         <<  2.* pixelInserter->GetCreator()->GetHalfDimension(0)/cm << Gateendl
      	      	  << ";Optical System pixel y dimension (cm) := "         <<  2.* pixelInserter->GetCreator()->GetHalfDimension(1)/cm << Gateendl
      	      	  << ";Optical System pixel z dimension (cm) := "         <<  2.* pixelInserter->GetCreator()->GetHalfDimension(2)/cm << Gateendl
      	      	  << ";Optical System pixel material := "                 <<  pixelInserter->GetCreator()->GetMaterialName() << Gateendl;

  m_headerFile  << ";\n";


}
